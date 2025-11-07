import streamlit as st
import requests
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, Part, FunctionDeclaration
from requests.auth import HTTPBasicAuth
import json
import os
from datetime import datetime, timedelta
import logging
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# ==== 0. STREAMLIT PAGE CONFIGURATION & LOGGING ====
# ==============================================================================

st.set_page_config(
    page_title="Darwinbox HR Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# ==== 1. CONFIGURATION & SECRETS ====
# ==============================================================================

# --- Vertex AI Project Configuration ---
PROJECT_ID = "sadproject2025"
LOCATION = "us-central1"

# --- Load Darwinbox Secrets from .env file ---
try:
    DOMAIN = os.getenv("DARWINBOX_DOMAIN")
    USERNAME = os.getenv("DARWINBOX_USERNAME")
    PASSWORD = os.getenv("DARWINBOX_PASSWORD")
    LEAVE_API_KEY = os.getenv("DARWINBOX_LEAVE_API_KEY")
    EMP_API_KEY = os.getenv("DARWINBOX_EMP_API_KEY")
    EMP_DATASET_KEY = os.getenv("DARWINBOX_EMP_DATASET_KEY")
    ATTENDANCE_API_KEY = os.getenv("DARWINBOX_ATTENDANCE_API_KEY")
    if not all([DOMAIN, USERNAME, PASSWORD, LEAVE_API_KEY, EMP_API_KEY, EMP_DATASET_KEY, ATTENDANCE_API_KEY]):
        raise ValueError("One or more DARWINBOX environment variables are missing.")
except Exception as e:
    st.error(f"🚨 Error loading credentials from .env file: {e}", icon="🔥")
    st.stop()

# ==============================================================================
# ==== 2. PYTHON FUNCTIONS (TOOLS) ====
# ==============================================================================
# These functions interact with the Darwinbox API. They return dictionaries.

def convert_date_format(date_string: str, from_format: str = '%Y-%m-%d', to_format: str = '%d-%m-%Y') -> str:
    return datetime.strptime(date_string, from_format).strftime(to_format)

def get_leave_report(employee_id: str, start_date: str, end_date: str) -> dict:
    """Retrieves approved leave records for a specific employee within a date range."""
    logger.info(f"Tool call: get_leave_report(emp_id={employee_id}, start={start_date}, end={end_date})")
    try:
        url = f"{DOMAIN}/leavesactionapi/leaveActionTakenLeaves"
        payload = { "api_key": LEAVE_API_KEY, "from": convert_date_format(start_date), "to": convert_date_format(end_date), "action": "2", "action_from": convert_date_format(start_date), "employee_no": [employee_id.strip()] }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def get_employee_info(employee_ids: list[str]) -> dict:
    """Gets profile data for one or more specific employees using their exact employee IDs."""
    logger.info(f"Tool call: get_employee_info(employee_ids={employee_ids})")
    try:
        url = f"{DOMAIN}/masterapi/employee"
        payload = {"api_key": EMP_API_KEY, "datasetKey": EMP_DATASET_KEY, "employee_ids": employee_ids}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def get_all_employees() -> dict:
    """Retrieves master data for ALL employees. Use this when you need to find an employee by name."""
    logger.info("Tool call: get_all_employees()")
    try:
        url = f"{DOMAIN}/masterapi/employee"
        payload = {"api_key": EMP_API_KEY, "datasetKey": EMP_DATASET_KEY}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def get_attendance_report(employee_ids: list[str], from_date: str, to_date: str) -> dict:
    """Retrieves daily attendance data for employees within a date range."""
    logger.info(f"Tool call: get_attendance_report(emp_ids={employee_ids}, from={from_date}, to={to_date})")
    try:
        url = f"{DOMAIN}/attendanceDataApi/DailyAttendanceRoster"
        payload = {"api_key": ATTENDANCE_API_KEY, "emp_number_list": employee_ids, "from_date": from_date, "to_date": to_date}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# ==============================================================================
# ==== 3. VERTEX AI MODEL CONFIGURATION ====
# ==============================================================================
@st.cache_resource
def setup_vertexai_model():
    """Initializes Vertex AI and the Generative Model with tools."""
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Define the tools using FunctionDeclaration
        tools = Tool.from_function_declarations([
            FunctionDeclaration(
                name="get_leave_report", description="Retrieves approved leave records for an employee within a date range.",
                parameters={"type": "object", "properties": {"employee_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["employee_id", "start_date", "end_date"]}
            ),
            FunctionDeclaration(
                name="get_employee_info", description="Gets profile data for employees using their exact employee IDs.",
                parameters={"type": "object", "properties": {"employee_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["employee_ids"]}
            ),
            FunctionDeclaration(
                name="get_all_employees", description="Retrieves master data for ALL employees. Use to find an employee by name.",
                parameters={"type": "object", "properties": {}}
            ),
            FunctionDeclaration(
                name="get_attendance_report", description="Retrieves daily attendance data for employees.",
                parameters={"type": "object", "properties": {"employee_ids": {"type": "array", "items": {"type": "string"}}, "from_date": {"type": "string"}, "to_date": {"type": "string"}}, "required": ["employee_ids", "from_date", "to_date"]}
            )
        ])

        today_str = datetime.now().strftime('%Y-%m-%d')
        system_prompt = f"""You are an AI HR assistant for Darwinbox. Today is {today_str}.
        Your goal is to answer user questions about employees, leaves, and attendance by using the available tools.
        **CRITICAL INSTRUCTIONS:**
        1.  If a user asks for information using an employee's **name** (e.g., "info for Sonali Garg"), you **MUST** first call `get_all_employees`. Then, search the results from that tool to find the employee's ID and other details to answer the question.
        2.  The tools `get_leave_report`, `get_employee_info`, and `get_attendance_report` only work with precise `employee_id`s.
        3.  Do not just dump raw data. Summarize the results from the tools in a clear, user-friendly format.
        """

        model = GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=system_prompt,
            tools=[tools]
        )
        return model
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI Model: {e}", icon="🔥")
        st.stop()

# ==============================================================================
# ==== 4. STREAMLIT APPLICATION MAIN LOGIC ====
# ==============================================================================
def main():
    st.title("🤖 Darwinbox HR Agent")
    st.caption(f"Powered by Vertex AI | Today: {datetime.now().strftime('%B %d, %Y')}")

    with st.sidebar:
        st.header("⚙️ Agent Details")
        st.info(f"**Project:** `{PROJECT_ID}`\n\n**Location:** `{LOCATION}`")
        with st.expander("Example Queries", expanded=True):
            st.markdown("- `Who is the manager for MMT6765?`\n- `Show me leaves for MMT6765 in Jan 2024`\n- `How many employees are there?`\n- `Attendance for EMP001 last week?`")

    # Initialize model and chat
    model = setup_vertexai_model()
    available_tools = {"get_leave_report": get_leave_report, "get_employee_info": get_employee_info, "get_all_employees": get_all_employees, "get_attendance_report": get_attendance_report}
    
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with Darwinbox HR tasks today?"}]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about an employee by name, leaves, attendance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # --- MANUAL FUNCTION CALLING LOOP (VERTEX AI SDK) ---
                    response = st.session_state.chat_session.send_message(prompt)
                    
                    # Check if the model decided to call a tool
                    if response.candidates[0].content.parts[0].function_call.name:
                        fn_call = response.candidates[0].content.parts[0].function_call
                        fn_name = fn_call.name
                        args = dict(fn_call.args)
                        
                        logger.info(f"Vertex AI function call: {fn_name} with args: {args}")

                        # Call the corresponding Python function
                        with st.spinner(f"Accessing tool: `{fn_name}`..."):
                            function_to_call = available_tools[fn_name]
                            function_response_data = function_to_call(**args)
                        
                        # Send the tool's response back to the model
                        with st.spinner("Processing tool results..."):
                            response = st.session_state.chat_session.send_message(
                                Part.from_function_response(
                                    name=fn_name,
                                    response={"content": function_response_data}
                                )
                            )
                        final_text = response.text
                    else:
                        # If no tool was called, use the direct text response
                        final_text = response.text

                    st.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})

                except Exception as e:
                    logger.error(f"Error during conversation: {e}\n{traceback.format_exc()}")
                    error_message = f"An unexpected error occurred: {str(e)}. Please try again."
                    st.error(error_message, icon="🔥")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()