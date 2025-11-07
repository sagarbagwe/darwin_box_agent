Here is a complete `README.md` file for your project, based on the `deploy.py` script, `.env` file, and `.gitignore` we've established.

This file is written in Markdown. You should save it as `README.md` in the root of your project folder.

-----

# Darwinbox HR Agent for Vertex AI

This project contains the code to deploy a comprehensive **Darwinbox HR Agent** to Google Cloud's Vertex AI Agent Engine. The agent is built with the Google Agent Development Kit (ADK) and comes pre-configured with 13 tools to interact with the Darwinbox HRMS API, handling queries for leaves, attendance, and employee data.

This project is designed to be secure and simple, using a single `deploy.py` script and a `.env` file for all configuration and secrets.

## 🌟 Features

  * **13 Pre-built Tools:** Ready to query leaves, attendance, and employee data.
  * **Secure by Design:** All API keys, passwords, and project IDs are loaded from a `.env` file and are **not** hardcoded in the script.
  * **Multi-Step Reasoning:** The agent is instructed to find an employee's ID from a name (`get_all_employees`) and then use that ID in other tools (e.g., `get_leave_balance`).
  * **Simple Deployment:** A single script handles agent definition, packaging, and deployment to Vertex AI.

## 📂 Project Structure

Your project directory should look like this:

```
/darwinbox-agent-project/
├── .env.example     # Template for your environment variables
├── .env             # (You create this) Your actual secrets file
├── .gitignore       # Ignores .env, venv, and cache files
├── deploy.py        # The main deployment script
└── requirements.txt   # Python dependencies
```

## 🚀 Setup and Deployment

Follow these steps to deploy the agent to your Google Cloud project.

### 1\. Prerequisites

  * Python 3.10 or later
  * A Google Cloud project with the **Vertex AI** and **Discovery Engine** APIs enabled.
  * A Google Cloud Storage bucket for staging. The script will use the name `gs://{YOUR_PROJECT_ID}-agent-engine-staging`.
  * `gcloud` CLI installed and authenticated on your local machine.

### 2\. Set Up a Python Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# 1. Create the virtual environment
python3 -m venv venv

# 2. Activate it (Linux/macOS)
source venv/bin/activate

# (For Windows)
# .\venv\Scripts\activate
```

### 3\. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

This is the most important step for securing your credentials.

1.  Make a copy of the template file. This new `.env` file is ignored by Git and will hold your secrets.

    ```bash
    cp .env.example .env
    ```

2.  Open the new `.env` file with a text editor.

3.  Fill in all the required values for your **Google Cloud project** and your **Darwinbox API credentials**. The comments in the file explain each variable.

### 5\. Authenticate with Google Cloud

Make sure your local SDK is authenticated to the Google Cloud project you specified in your `.env` file.

```bash
# 1. Log in to your Google account
gcloud auth login

# 2. Set up Application Default Credentials
gcloud auth application-default login

# 3. Set your project as the default for the gcloud CLI
gcloud config set project YOUR_PROJECT_ID
```

*(Replace `YOUR_PROJECT_ID` with the value from your `.env` file).*

### 6\. Run the Deployment

With your virtual environment active and your `.env` file populated, you are ready to deploy.

Run the `deploy.py` script:

```bash
python deploy.py
```

The script will:

1.  Load and validate all configuration from your `.env` file.
2.  Initialize the Vertex AI client.
3.  Define the 13 tools and the agent's system prompt.
4.  Package the agent and deploy it to Vertex AI Agent Engine.

This final step can take **15-20 minutes** to complete. If successful, the script will print the deployed agent's resource name.

-----

## ➡️ Next Steps: Registering the Agent

After the `deploy.py` script succeeds, your agent is **deployed** as a "Reasoning Engine."

To make it usable in a chat application, you must **register** it with a Vertex AI Search (Discovery Engine) "Engine". You can do this using the `curl` command you've used before, which links your Discovery Engine to this new Reasoning Engine ID.