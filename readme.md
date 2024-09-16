# AWS Deployment Guide

This guide walks you through the process of deploying your application on an AWS EC2 instance.

## 1. Launch EC2 Instance

1. Go to the Amazon AWS console and search for "EC2".
2. Click on "Launch Instance".
3. Provide a name for your instance.

![Step 1](STEPS/1.png)

4. Select the instance type that suits your needs.

![Step 2](STEPS/2.png)

## 2. Configure Network Settings

1. Click on "Edit" in the Network Settings section.

![Step 3](STEPS/3.png)

2. Click on "Add security group rule".

![Step 4](STEPS/4.png)

3. Add port range 8501 and set the source type to "Anywhere".
4. Click on "Launch Instance".

![Step 5](STEPS/5.png)

## 3. Connect to Your Instance

1. Once the instance is launched, click on the instance ID.

![Step 6](STEPS/6.png)

2. Click on "Connect".

![Step 7](STEPS/7.png)

## 4. Set Up the Environment

1. In the CLI, change to superuser:
   ```
   sudo su
   ```

2. Install git:
   ```
   yum install git
   ```

![Step 8](STEPS/8.png)

3. Install Python3-pip:
   ```
   yum install python3-pip
   ```

![Step 9](STEPS/9.png)

4. Clone your repository:
   ```
   git clone <your_repo_link>
   ```

5. Create a Python virtual environment:
   ```
   python3 -m venv myenv
   ```

![Step 10](STEPS/10.png)

6. Install necessary libraries:
   ```
   pip install --no-cache-dir streamlit sentence-transformers pinecone-client openai==0.28 pdfplumber
   ```

![Step 11](STEPS/11.png)

## 5. Configure and Run Your Application

1. Change directory to your repo:
   ```
   cd <directory_name>
   ```

2. Edit your Python file to add your API key:
   ```
   nano <Your_pythonfile.py>
   ```

![Step 12](STEPS/12.png)

3. Run your script:
   ```
   streamlit run <Your_pythonfile.py>
   ```

4. For continuous running of the instance even after closing the tab:
   ```
   nohup python3 -m streamlit run <Your_pythonfile.py> &
   ```

![Step 13](STEPS/13.png)

Congratulations! Your application should now be deployed and running on AWS EC2.