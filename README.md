# Running the project

## Environment Setup - Python

1. Create python virtual environment
   ```sh
   python3.11 -m venv myenv
   ```

2. Activate the environment
   ```sh
   source myenv/bin/activate
   ```

3. Verify latest version of pip is installed
   ```sh
   pip install --upgrade pip
   ```

4. Install required python packages
   ```sh
   pip install -r requirements.txt
   ```

# AWS Deployment Guide

This guide walks you through the process of deploying your application on an AWS EC2 instance.

## 1. Launch EC2 Instance

1. Go to the Amazon AWS console and search for "EC2".
2. Click on "Launch Instance".
3. Provide a name for your instance.

<details>
<summary>View Screenshot</summary>

![Step 1](11_deployment_steps/1.jpg)

</details>

4. Select the instance type that suits your needs.

<details>
<summary>View Screenshot</summary>

![Step 2](11_deployment_steps/2.jpg)

</details>

## 2. Configure Network Settings

1. Click on "Edit" in the Network Settings section.

<details>
<summary>View Screenshot</summary>

![Step 3](11_deployment_steps/3.jpg)

</details>

2. Click on "Add security group rule".

<details>
<summary>View Screenshot</summary>

![Step 4](11_deployment_steps/4.jpg)

</details>

3. Add port range 8501 and set the source type to "Anywhere".
4. Click on "Launch Instance".

<details>
<summary>View Screenshot</summary>

![Step 5](11_deployment_steps/5.jpg)

</details>

## 3. Connect to Your Instance

1. Once the instance is launched, click on the instance ID.

<details>
<summary>View Screenshot</summary>

![Step 6](11_deployment_steps/6.jpg)

</details>

2. Click on "Connect".

<details>
<summary>View Screenshot</summary>

![Step 7](11_deployment_steps/7.jpg)

</details>

## 4. Set Up the Environment

1. In the CLI, change to superuser:
   ```
   sudo su
   ```

2. Install git:
   ```
   yum install git
   ```

<details>
<summary>View Screenshot</summary>

![Step 8](11_deployment_steps/8.jpg)

</details>

3. Install Python3-pip:
   ```
   yum install python3-pip
   ```

<details>
<summary>View Screenshot</summary>

![Step 9](11_deployment_steps/9.jpg)

</details>

4. Clone your repository:
   ```
   git clone <your_repo_link>
   ```

5. Create a Python virtual environment:
   ```
   python3 -m venv myenv
   ```

<details>
<summary>View Screenshot</summary>

![Step 10](11_deployment_steps/10.jpg)

</details>

6. Install necessary libraries:
   ```
   pip install --no-cache-dir streamlit sentence-transformers pinecone-client openai==0.28 pdfplumber
   ```

<details>
<summary>View Screenshot</summary>

![Step 11](11_deployment_steps/11.jpg)

</details>

## 5. Configure and Run Your Application

1. Change directory to your repo:
   ```
   cd <directory_name>
   ```

2. Edit your Python file to add your API key:
   ```
   nano <Your_pythonfile.py>
   ```

<details>
<summary>View Screenshot</summary>

![Step 12](11_deployment_steps/12.jpg)

</details>

3. Run your script:
   ```
   streamlit run <Your_pythonfile.py>
   ```

4. For continuous running of the instance even after closing the tab:
   ```
   nohup python3 -m streamlit run <Your_pythonfile.py> &
   ```

<details>
<summary>View Screenshot</summary>

![Step 13](11_deployment_steps/13.jpg)

</details>

Congratulations! Your application should now be deployed and running on AWS EC2.

