
# RAG AI App Readme

Welcome to the RAG AI app repository! This app integrates with various AI models to provide insightful responses based on the provided context, making it an excellent tool for exploring AI's capabilities. The app also features integration with Gmail for fetching and processing emails, alongside functionality for handling different types of documents.

This RAG pipeline uses Llamaindex tooling, Qdrant as a vectorDB, HuggingFace for embedding (bge-small), and your choice of LLM for the open context response. You can modify these things but I found this combination to be relatively simple to implement, well documented, and opensource. If you run your QdrantDB locally in docker or selfhosted, you use huggingface, and run your LLM locally in LMStudio or equivalent, this entire rag pipeline is closed to your local environment. I have not implemented any kind of logging systems or external tracking. Certainly as this is built out I would like to optionally enable logging for internal purposes but as of now 3/15/24, this could be run entirely locally!

## Contribution Guidelines

I welcome contributions from the community! If you wish to contribute:

1. Please create an issue for any bugs, feature requests, or other enhancements. This helps me keep track of what needs attention.
2. Feel free to fork the repository and create pull requests. Make sure to create your branch from `main`. For bugs, features, refactors, or anything new, use a descriptive branch name, e.g., `git checkout -b feature/add-new-integration`.
3. All pull requests require approval before being merged. This ensures code quality and consistency.
4. In the future, I plan to introduce a `develop` branch as the main staging area. Changes will be merged there first before making their way to `main`.

## Getting Started

To get started with the RAG AI app:

1. Clone the repository to your local machine.
2. Copy the example environment file and configure it with your API keys and other settings:
   `$ cp example.env .env`
3. Update the `.env` file with your Qdrant API key and URL. Alternatively, update `app.py` to use local memory for development:
   `qdrant_client.QdrantClient(":memory:")`
4. Add API keys for OpenAI, Anthropic, and Mistral in the `.env` file as needed.
5. Install required dependencies:
   `$ pip install -r requirements.txt`
6. Serve the app locally using Streamlit:
   `$ streamlit run app.py`

### Email Integration

For email functionality, you'll need to:

* Authorize your Google account and enable GCP Cloud Services for Gmail (full access) and read-only access for Contacts, Tasks, and Calendar. Additional features will be introduced in future updates.
* Set up OAuth2 flow with a redirect to localhost:3000.
* Download the credentials from your GCP app and save them as `myaiCred.json` in the root directory.
  * **IMPORTANT**: This is in the `.gitignore`, so if you change the file name, please update the `.gitignore` so you do not publish your gcp credentials
  * Please also update  `auth.py` to use the correct file name (it is hard coded currently)
* When the app successfully authenticates it generates a local token.json file at the root of the project. This contains some auth details including the auth token and refresh tokens
  * If you need to delete the `token.json` you can and reauthenticate your application. The `token.json` is in the `.gitgnore` so please also update any references if changing the name of this.
* The app creates an epoch timestamp in seconds, used for querying new emails since the last run.
  * This is preserved in the streamlit session and in a generated json file `last_query_timestamp.json` which is also in the `.gitignore`)
* Adjust the maximum email retrieval limit and categories as needed in `mail_manager.py`.

### Adding Documents

When adding documents:

* Collections in Qdrant are specific to the file type (e.g., web, pdf, csv, docx).
* Use the appropriate reader and parser based on the file type. For web pages, specify the crawl depth to control how deeply the app should crawl linked pages.
* The conversation context is passed along with retrieved documents to the AI model for generating responses.

## Notes

* The Streamlit app manages session states and stashes the last email query timestamp for efficient email fetching.
* Feel free to adjust the timestamp manually for specific backfill start points, keeping in mind the maximum retrieval limit.
* If you want to use a local model, you should be able to do this, I have used LM studio in development, you should initialize the streamlit app with a 'local' llm. This will initialize the rag tool using the follwing information (default port and key for LMStudio LLMs running locally).
  * llm = OpenAI(model="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")
  * You may need to play with token context limits in your local model or modify the app.py to limit them, use the LlamaIndex Settings to modify paramters as needed

## Hope to See Your Contributions!

This project is open to contributions, and we're excited to see how it grows with the community's input. Whether it's bug fixes, new features, or improvements, your contributions are welcome!
