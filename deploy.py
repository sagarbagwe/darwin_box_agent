#!/usr/bin/env python3
"""
Vertex AI Agent Deployment Script for Darwinbox HRMS.

This script loads configuration from a .env file, defines all agent tools,
and deploys the agent to Vertex AI Agent Engine.
"""

# --- Python Standard Library Imports ---
import os
import sys
import json
import logging
import traceback
from datetime import datetime
from requests.auth import HTTPBasicAuth

# --- Third-Party Imports ---
import requests
from dotenv import load_dotenv  # Import dotenv

# --- Vertex AI and ADK Imports ---
import vertexai
from google.adk.agents import LlmAgent
from vertexai.preview.reasoning_engines import AdkApp
from vertexai import agent_engines

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# ==== 1. CONFIGURATION & SECRETS (NOW LOADED FROM .ENV) ====
# ==============================================================================
print("🚀 Starting Darwinbox HR Agent deployment process...")

# Load environment variables from .env file
load_dotenv()

def get_env_variable(var_name: str, is_secret: bool = False) -> str:
    """Helper function to get an environment variable, validate it, and log."""
    value = os.environ.get(var_name)
    if not value:
        logger.error(f"FATAL: Environment variable '{var_name}' is not set. Please check your .env file.")
        sys.exit(f"Error: Missing required environment variable: {var_name}")
    
    if is_secret:
        logger.info(f"Loaded config: {var_name} = '********'")
    else:
        logger.info(f"Loaded config: {var_name} = '{value}'")
    return value

# --- Vertex AI Project Configuration ---
PROJECT_ID = get_env_variable("PROJECT_ID")
LOCATION = get_env_variable("LOCATION")
GEMINI_MODEL = get_env_variable("GEMINI_MODEL")

# --- Darwinbox API Credentials ---
DOMAIN = get_env_variable("DARWINBOX_DOMAIN")
USERNAME = get_env_variable("DARWINBOX_USERNAME")
PASSWORD = get_env_variable("DARWINBOX_PASSWORD", is_secret=True)

# --- Darwinbox API Keys (Consolidated) ---
LEAVE_KEYS = {
    "report": get_env_variable("LEAVE_REPORT_KEY", is_secret=True),
    "action": get_env_variable("LEAVE_ACTION_KEY", is_secret=True),
    "holiday": get_env_variable("LEAVE_HOLIDAY_KEY", is_secret=True),
    "balance": get_env_variable("LEAVE_BALANCE_KEY", is_secret=True),
    "encashment": get_env_variable("LEAVE_ENCASHMENT_KEY", is_secret=True),
    "import": get_env_variable("LEAVE_IMPORT_KEY", is_secret=True)
}

ATTENDANCE_KEYS = {
    "daily_roster": get_env_variable("ATTENDANCE_DAILY_ROSTER_KEY", is_secret=True),
    "daily_status": get_env_variable("ATTD_DAILY_STATUS_KEY", is_secret=True),
    "punches": get_env_variable("ATTENDANCE_PUNCHES_KEY", is_secret=True),
    "monthly": get_env_variable("ATTENDANCE_MONTHLY_KEY", is_secret=True),
    "datewise_roster": get_env_variable("ATTD_DATEWISE_ROSTER_KEY", is_secret=True),
    "compoff": get_env_variable("ATTD_COMPOFF_KEY", is_secret=True),
    "timesheet": get_env_variable("ATTENDANCE_TIMESHEET_KEY", is_secret=True),
    "timesheet_datewise": get_env_variable("ATTENDANCE_TIMESHEET_DATEWISE_KEY", is_secret=True),
    "overtime_datewise": get_env_variable("ATTENDANCE_OVERTIME_DATEWISE_KEY", is_secret=True)
}

EMP_API_KEY = get_env_variable("EMP_API_KEY", is_secret=True)
EMP_DATASET_KEY = get_env_variable("EMP_DATASET_KEY", is_secret=True)

# --- Staging Bucket ---
# A GCS bucket is required for staging the agent artifacts during deployment.
# This logic remains the same, but it now uses the loaded PROJECT_ID.
STAGING_BUCKET = f"gs://{PROJECT_ID}-agent-engine-staging"

# --- Initialize Vertex AI ---
print(f"✅ Initializing Vertex AI for project '{PROJECT_ID}' in '{LOCATION}'...")
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
print("✅ Configuration loaded and Vertex AI initialized.")

# ==============================================================================
# ==== 2. UTILITY FUNCTIONS ====
# ==============================================================================
# (No changes needed in this section)

def validate_date_format(date_string: str) -> bool:
    """Validate if date is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def convert_date_format(date_string: str, from_format: str = '%Y-%m-%d', to_format: str = '%d-%m-%Y') -> str:
    """Convert date from one format to another"""
    try:
        return datetime.strptime(date_string, from_format).strftime(to_format)
    except ValueError as e:
        logger.error(f"Date conversion failed: {e}")
        raise ValueError(f"Invalid date format: {date_string}")

def validate_employee_id(employee_id: str) -> bool:
    """Validate employee ID is not empty"""
    return employee_id and isinstance(employee_id, str) and len(employee_id.strip()) >= 1

def _handle_api_error(e: Exception, tool_name: str) -> str:
    """Generic error handler for API calls."""
    logger.error(f"Error in {tool_name}: {e}\n{traceback.format_exc()}")
    if isinstance(e, requests.exceptions.HTTPError):
        return json.dumps({"error": f"API Error: {e.response.status_code}", "details": e.response.text[:500]})
    if isinstance(e, requests.exceptions.Timeout):
        return json.dumps({"error": "API Error: Request timed out"})
    return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

# ==============================================================================
# ==== 3. PYTHON FUNCTIONS (TOOLS) ====
# ==============================================================================
# (No changes needed in this section, as all functions refer to the global
# config variables which are now loaded from the .env file.)
print("🛠️ Defining the Darwinbox tools...")

# ==== LEAVE MANAGEMENT TOOLS ====

def get_leave_report(employee_no: str, start_date: str, end_date: str) -> str:
    """Get leave records for an employee between start_date and end_date in YYYY-MM-DD format"""
    logger.info(f"Tool call: get_leave_report(emp={employee_no}, start={start_date}, end={end_date})")
    try:
        if not all([validate_employee_id(employee_no), validate_date_format(start_date), validate_date_format(end_date)]):
            return json.dumps({"error": "Invalid input parameters. Check employee_no and date formats (YYYY-MM-DD)."})
        
        url = f"{DOMAIN}/leavesactionapi/leaveActionTakenLeaves"
        payload = {
            "api_key": LEAVE_KEYS["report"],
            "from": convert_date_format(start_date),
            "to": convert_date_format(end_date),
            "action": "2",
            "action_from": convert_date_format(start_date),
            "employee_no": [employee_no.strip()]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "get_leave_report")

def get_leave_balance(employee_nos: list[str], leave_names: list[str] = None) -> str:
    """Get leave balance for one or more employees. employee_nos must be a list of strings."""
    logger.info(f"Tool call: get_leave_balance(emps={employee_nos}, leaves={leave_names})")
    try:
        url = f"{DOMAIN}/leavesactionapi/leavebalance"
        payload = {
            "api_key": LEAVE_KEYS["balance"],
            "ignore_rounding": "1",
            "employee_nos": [str(no).strip() for no in employee_nos],
            "leave_names": leave_names or []
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "get_leave_balance")

def update_leave_status(employee_no: str, leave_id: str, action: str, manager_message: str = "") -> str:
    """Approve, Reject, or Revoke a specific leave request. 'action' must be one of: Approved, Rejected, Revoked"""
    logger.info(f"Tool call: update_leave_status(emp={employee_no}, leave_id={leave_id}, action={action})")
    try:
        capitalized_action = action.capitalize()
        if capitalized_action not in ["Approved", "Rejected", "Revoked"]:
            return json.dumps({"error": f"Invalid action: {action}. Must be 'Approved', 'Rejected', or 'Revoked'."})
        
        url = f"{DOMAIN}/leavesactionapi/leaveaction"
        payload = {
            "api_key": LEAVE_KEYS["action"],
            "employee_no": employee_no.strip(),
            "leave_id": leave_id,
            "action": capitalized_action,
            "manager_message": manager_message
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "update_leave_status")

def apply_for_leave(employee_no: str, leave_name: str, start_date: str, end_date: str, 
                   is_half_day: bool = False, is_first_half: bool = True, 
                   is_paid: bool = True, message: str = "Applied via AI") -> str:
    """Apply for a new leave on behalf of an employee. Dates should be in YYYY-MM-DD format"""
    logger.info(f"Tool call: apply_for_leave(emp={employee_no}, leave={leave_name}, start={start_date}, end={end_date})")
    try:
        if not all([validate_employee_id(employee_no), validate_date_format(start_date), validate_date_format(end_date)]):
            return json.dumps({"error": "Invalid input parameters. Check employee_no and date formats (YYYY-MM-DD)."})
        
        leave_data = {
            "employee_no": employee_no.strip(),
            "leave_name": leave_name,
            "message": message,
            "from_date": convert_date_format(start_date, to_format='%d-%m-%Y'),
            "to_date": convert_date_format(end_date, to_format='%d-%m-%Y'),
            "is_half_day": "yes" if is_half_day else "no",
            "is_paid_or_unpaid": "paid" if is_paid else "unpaid",
            "revoke_leave": "no"
        }
        
        if is_half_day:
            leave_data["is_firsthalf_secondhalf"] = "1" if is_first_half else "2"
        
        url = f"{DOMAIN}/leavesactionapi/importleave"
        payload = {"api_key": LEAVE_KEYS["import"], "data": [leave_data]}
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "apply_for_leave")

def get_holiday_list(employee_no: str, year: str = None) -> str:
    """Get the holiday calendar for a specific employee and year. If 'year' not provided, uses current year."""
    logger.info(f"Tool call: get_holiday_list(emp={employee_no}, year={year})")
    try:
        year = year or str(datetime.now().year)
        if not validate_employee_id(employee_no):
             return json.dumps({"error": "Invalid employee_no."})

        url = f"{DOMAIN}/leavesactionapi/holidaylist"
        payload = {"api_key": LEAVE_KEYS["holiday"], "employee_no": employee_no.strip(), "year": year}
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "get_holiday_list")

def get_leave_encashment_details(employee_no: str, start_date: str, end_date: str) -> str:
    """Get leave encashment details for an employee. Dates must be in YYYY-MM-DD format."""
    logger.info(f"Tool call: get_leave_encashment_details(emp={employee_no}, start={start_date}, end={end_date})")
    try:
        if not all([validate_employee_id(employee_no), validate_date_format(start_date), validate_date_format(end_date)]):
            return json.dumps({"error": "Invalid input parameters. Check employee_no and date formats (YYYY-MM-DD)."})
        
        from_datetime = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d-%m-%Y 00:00:00')
        to_datetime = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d-%m-%Y 23:59:59')
        
        url = f"{DOMAIN}/leavesactionapi/encashmentDetails"
        payload = {
            "api_key": LEAVE_KEYS["encashment"],
            "from": from_datetime,
            "to": to_datetime,
            "employee_no": [employee_no.strip()]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return _handle_api_error(e, "get_leave_encashment_details")

# ==== ATTENDANCE TOOLS ====

def get_daily_attendance_status(employee_ids: list[str], attendance_date: str) -> str:
    """Fetch daily attendance status (Present/Absent, timings) for one or more employees for a SINGLE date. Date in YYYY-MM-DD format."""
    logger.info(f"Tool call: get_daily_attendance_status(emps={employee_ids}, date={attendance_date})")
    try:
        if not validate_date_format(attendance_date):
             return json.dumps({"error": "Invalid date format. Please use YYYY-MM-DD."})

        url = f"{DOMAIN}/AttendanceDataApi/daily"
        payload = {
            "api_key": ATTENDANCE_KEYS["daily_status"],
            "emp-number_list": [str(e).strip() for e in employee_ids],
            "attendance_date": convert_date_format(attendance_date) # Convert to dd-mm-yyyy
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except Exception as e:
        return _handle_api_error(e, "get_daily_attendance_status")

def get_daily_attendance_roster(employee_ids: list[str], from_date: str, to_date: str) -> str:
    """Get daily attendance roster (shift, status, hours) for one or more employees for a date range. Dates in YYYY-MM-DD format."""
    logger.info(f"Tool call: get_daily_attendance_roster(emps={employee_ids}, from={from_date}, to={to_date})")
    try:
        url = f"{DOMAIN}/attendanceDataApi/DailyAttendanceRoster"
        payload = {
            "api_key": ATTENDANCE_KEYS["daily_roster"],
            "emp_number_list": [str(e).strip() for e in employee_ids],
            "from_date": from_date,
            "to_date": to_date
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except Exception as e:
        return _handle_api_error(e, "get_daily_attendance_roster")

def get_attendance_punches(employee_ids: list[str], from_date: str, to_date: str) -> str:
    """Get raw attendance punch-in/out records for one or more employees. Dates in YYYY-MM-DD format"""
    logger.info(f"Tool call: get_attendance_punches(emps={employee_ids}, from={from_date}, to={to_date})")
    try:
        url = f"{DOMAIN}/AttendancePunchesApi"
        payload = {
            "api_key": ATTENDANCE_KEYS["punches"],
            "emp_number_list": [str(e).strip() for e in employee_ids],
            "from_date": convert_date_format(from_date),
            "to_date": convert_date_format(to_date)
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except Exception as e:
        return _handle_api_error(e, "get_attendance_punches")

def get_monthly_attendance(employee_ids: list[str], month_year: str) -> str:
    """Get monthly attendance summary. 'month_year' MUST be in YYYY-MM format (e.g., '2025-10')."""
    logger.info(f"Tool call: get_monthly_attendance(emps={employee_ids}, month={month_year})")
    try:
        # Validate YYYY-MM format
        datetime.strptime(month_year, '%Y-%m')

        url = f"{DOMAIN}/AttendanceDataApi/monthly"
        payload = {
            "api_key": ATTENDANCE_KEYS["monthly"],
            "emp_number_list": [str(e).strip() for e in employee_ids],
            "month": month_year
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except ValueError:
        return json.dumps({"error": "Invalid month_year format. Must be YYYY-MM."})
    except Exception as e:
        return _handle_api_error(e, "get_monthly_attendance")

def get_timesheet_datewise(employee_ids: list[str], from_date: str, to_date: str) -> str:
    """Get employee timesheet data (projects, tasks) datewise. Dates must be in YYYY-MM-DD format."""
    logger.info(f"Tool call: get_timesheet_datewise(emps={employee_ids}, from={from_date}, to={to_date})")
    try:
        from_date_converted = convert_date_format(from_date)
        to_date_converted = convert_date_format(to_date)
        
        url = f"{DOMAIN}/attendanceDataApi/timesheetdatewise"
        payload = {
            "api_key": ATTENDANCE_KEYS["timesheet_datewise"],
            "from": from_date_converted,
            "to": to_date_converted,
            "emp_number_list": [str(e).strip() for e in employee_ids]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except Exception as e:
        return _handle_api_error(e, "get_timesheet_datewise")

def get_overtime_datewise(employee_ids: list[str], from_date: str, to_date: str) -> str:
    """Get employee overtime data for a date range. Dates must be in YYYY-MM-DD format."""
    logger.info(f"Tool call: get_overtime_datewise(emps={employee_ids}, from={from_date}, to={to_date})")
    try:
        from_date_converted = convert_date_format(from_date)
        to_date_converted = convert_date_format(to_date)

        url = f"{DOMAIN}/attendanceDataApi/getOverTimeDatewise"
        payload = {
            "api_key": ATTENDANCE_KEYS["overtime_datewise"],
            "from": from_date_converted,
            "to": to_date_converted,
            "emp_number_list": [str(e).strip() for e in employee_ids]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return json.dumps({"status": "success", "data": response.json()})
    except Exception as e:
        return _handle_api_error(e, "get_overtime_datewise")

# ==== EMPLOYEE TOOLS ====

def get_all_employees() -> str:
    """Get complete employee database. Use this to find an employee's ID ('employee_number') when the user asks by name."""
    logger.info("Tool call: get_all_employees()")
    try:
        url = f"{DOMAIN}/masterapi/employee"
        payload = {"api_key": EMP_API_KEY, "datasetKey": EMP_DATASET_KEY}
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                               auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=60)
        response.raise_for_status()
        api_data = response.json()
        count = len(api_data.get("data", [])) if isinstance(api_data, dict) else len(api_data)
        return json.dumps({"status": "success", "employee_count": count, "data": api_data})
    except Exception as e:
        return _handle_api_error(e, "get_all_employees")


# ==============================================================================
# ==== 4. DEFINE YOUR AGENT ====
# ==============================================================================
# (No changes needed in this section)
print("🤖 Defining the Darwinbox Agent logic...")

# --- Consolidate all tools ---
darwinbox_tools = [
    # Leave Tools
    get_leave_report, 
    get_leave_balance, 
    apply_for_leave, 
    update_leave_status,
    get_holiday_list, 
    get_leave_encashment_details,
    # Attendance Tools
    get_daily_attendance_status,
    get_daily_attendance_roster, 
    get_attendance_punches, 
    get_monthly_attendance, 
    get_timesheet_datewise, 
    get_overtime_datewise,
    # Employee Tools
    get_all_employees
]

# --- Define System Prompt ---
today = datetime.now().strftime('%Y-%m-%d')

system_prompt = f"""You are Darwin, an AI HR Assistant for Darwinbox HRMS. Today is {today}.
Your primary function is to use the available tools to answer user questions about employee leaves, profiles, and attendance.

**CRITICAL INSTRUCTIONS:**

1.  **ID vs. Name Distinction:**
    * Almost all tools (leave, attendance) require a precise `employee_no` or `employee_ids` list. They DO NOT work with employee names.
    * The ONLY exception is `get_all_employees()`.

2.  **Multi-Step Process for Names:**
    * If a user asks about an employee by **name** (e.g., "What is David ABC's leave balance?" or "Show attendance for Sonli Garg"), you MUST follow this two-step process:
    * **Step 1:** Call `get_all_employees()` to retrieve the complete employee list.
    * **Step 2:** Search the retrieved JSON data for the requested name to find their exact `employee_number`.
    * **Step 3:** Call the appropriate tool (e.g., `get_leave_balance`) using the `employee_number` you found.
    * DO NOT ask the user for the ID if they provide a name. Find it yourself.

3.  **Date Handling:**
    * Today's date is **{today}**.
    * Convert all relative date queries ("last week", "this month", "last month", "yesterday") into absolute `YYYY-MM-DD` date ranges based on today's date.
    * Example for "last month": If today is 2025-11-06, "last month" is `from_date: "2025-10-01"` and `to_date: "2025-10-31"`.
    * Example for "last week": If today is 2025-11-06 (Thursday), "last week" is Monday `2025-10-27` to Sunday `2025-11-02`.
    * All dates provided to tools MUST be in `YYYY-MM-DD` format.

4.  **Attendance Tool Selection Logic:**
    * For a **single day's** status (P/A, timings) -> `get_daily_attendance_status()`.
    * For a general "daily roster" over a **date range** -> `get_daily_attendance_roster()`.
    * For a detailed **timesheet** with project/activity data -> `get_timesheet_datewise()`.
    * For raw **punch-in/out** times -> `get_attendance_punches()`.
    * For a **monthly summary** -> `get_monthly_attendance()` (Note: `month_year` param MUST be "YYYY-MM").
    * For **overtime** data -> `get_overtime_datewise()`.

5.  **Summarize Results:**
    * Do not just dump raw JSON. Present the information from the tools in a clear, user-friendly format (e.g., a summary sentence or a markdown table).
    * If a tool returns an error, state the error clearly and suggest a fix (e.g., "I couldn't find an employee with that ID. Please check the ID and try again.").
"""

# --- Create the LlmAgent ---
darwinbox_agent = LlmAgent(
    name="darwinbox_hr_agent",
    model=GEMINI_MODEL,
    instruction=system_prompt,
    tools=darwinbox_tools
)

print("✅ Agent definition complete.")

# ==============================================================================
# ==== 5. PACKAGE AND DEPLOY ====
# ==============================================================================
# (No changes needed in this section)
if __name__ == "__main__":
    print("📦 Packaging agent with AdkApp...")
    
    # Enable tracing for better debugging in the Vertex AI console
    app = AdkApp(
        agent=darwinbox_agent,
        enable_tracing=True,
    )

    # Define the Python requirements for the deployed agent
    deployment_requirements = [
        "google-cloud-aiplatform[agent_engines,adk]>=1.55.0",
        "google-adk>=0.1.0",
        "requests>=2.31.0",
        # python-dotenv is NOT needed in the deployed environment,
        # as it's only used for loading the local .env file.
    ]

    print("🛰️ Deploying to Vertex AI Agent Engine... (This may take 15-20 minutes)")
    try:
        # Create (and deploy) the agent
        remote_app = agent_engines.create(
            app,
            display_name="darwinbox-hr-agent-full",
            description="Comprehensive HR agent for Darwinbox (Leave, Attendance, and Employee Directory)",
            requirements=deployment_requirements,
            # Pass agent_id to update an existing agent
            # agent_id="my-existing-agent-id"
        )
        
        print("\n🎉 Deployment successful!")
        print("Agent Resource Name:", remote_app.resource_name)
        print("You can now interact with your agent via the API or the Google Cloud Console.")
        print(remote_app)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}", file=sys.stderr)
        print("Please check your permissions, project configuration, and the GCS staging bucket.")
        sys.exit(1)