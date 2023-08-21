
## Chat with PDF

Upload your document and ask your queries related to document to get your answer. This application uses langchain to search within your document and OPENAI's gpt-3.5-turbo to process and response your queries.

## Run Locally

Export OPENAI API KEY

```bash
  export OPENAI_API_KEY=$YOUR_OPEN_API_KEY
```

Clone the project

```bash
  git clone git@github.com:SubinMaharjan/chat_pdf.git
```

Go to the project directory

```bash
  cd chat_pdf
```

Create and activate virtual enviroment
```bash
  python -m venv project
  source project/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the application

```bash
  streamlit run app_v1.py
```

