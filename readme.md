
# PDF Questioning Tool

This tool enables users to ask questions directly from PDF files.

## Steps to Run:

1. **Install the Requirements**:
   - Use `pip` to install the necessary packages by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Set Environment Variables**:
   - Configure the LLM (Language Model) variables in the `.env` file. Currently, only OpenAI is supported.

3. **Run the Streamlit Application**:
   - Start the app by executing:
     ```bash
     streamlit run chat_web_st.py
     ```

4. **Using the Application**:
   - Once the app is running, upload your PDF document.
   - Press the "Process" button and wait for 1-2 minutes, depending on the length of the file, as it undergoes vectorization.
   - After the processing is complete, you can start asking questions related to the content of the PDF.

## Notes:
- Ensure that your `.env` file contains valid OpenAI API keys.
- The processing time may vary based on the size of the PDF file.

Enjoy querying your PDF files with ease!
