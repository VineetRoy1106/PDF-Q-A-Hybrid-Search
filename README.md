# PDF-Q-A-Hybrid-Search

App-    https://vroy-rag.streamlit.app/

Here's the formatted README for you to copy and use:

---


## **Introduction**
This Streamlit application combines powerful tools such as Pinecone, Groq, and HuggingFace embeddings to provide a hybrid search and question-answering (QA) experience based on uploaded PDF documents. The app allows users to upload PDFs, process their content, and query the data in natural language.

### **Core Features**
1. **PDF Upload and Content Processing**:
   - Users can upload multiple PDF files to the application.
   - The app processes these PDFs and extracts text content for search and QA.

2. **Hybrid Search with Pinecone**:
   - Combines dense embeddings (via HuggingFace) and sparse representations (via BM25) for accurate and efficient retrieval of relevant content.

3. **Question Answering with Groq**:
   - Leverages Groq’s large language models (LLMs) to generate accurate answers to user queries based on retrieved content.

4. **Interactive User Interface**:
   - Built with Streamlit for a simple and intuitive user experience.
  
  ![Query1](https://github.com/user-attachments/assets/cec71e83-00fc-4ad9-b069-1b351bd89666)


---

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or higher
- Pip package manager
- Pinecone and Groq API keys

### **Environment Variables**
The application uses the following environment variables, which can be stored in a `.env` file or managed securely with Streamlit secrets:

- `PINECONE_API_KEY`: API key for Pinecone
- `GROQ_API_KEY`: API key for Groq

### **Installation Steps**

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys:**
   - Create a `.env` file in the project root or use Streamlit secrets.
   - Add the following keys:
     ```plaintext
     PINECONE_API_KEY=<your_pinecone_api_key>
     GROQ_API_KEY=<your_groq_api_key>
     ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## **Code Walkthrough**

### **1. Imports and Setup**
The app imports necessary libraries for:
- **PDF loading and processing:** PyPDFLoader
- **Search and embeddings:** PineconeHybridSearchRetriever, HuggingFaceEmbeddings, BM25Encoder
- **QA Chain and LLMs:** LangChain and Groq integration
- **Environment variable management:** dotenv
- **UI:** Streamlit

### **2. Initializing Pinecone**
- The code checks if the specified Pinecone index exists; if not, it creates one.
- A hybrid search index is prepared using Pinecone's dense and sparse retrieval capabilities.

### **3. PDF Upload and Content Extraction**
- Users upload PDFs using Streamlit's `file_uploader` widget.
- The app uses `PyPDFLoader` to extract and store text content from the uploaded files.

### **4. Hybrid Search Retriever**
- Sparse and dense encoders are applied to the extracted text.
- The text content is indexed in Pinecone for efficient hybrid search retrieval.

### **5. Question Answering (QA)**
- The app initializes a Groq LLM (e.g., Llama3-8b-8192) for generating answers.
- Using LangChain's QA chain, the app retrieves the most relevant documents and formulates a natural language response to the user's query.

### **6. User Query and Output**
- Users input their query through a text box.
- The app retrieves relevant content, processes it using Groq's LLM, and displays the answer.

---

## **How to Use**
1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload PDFs:**
   - Drag and drop one or more PDF files into the upload section.

3. **Ask a Question:**
   - Type a question related to the content of the uploaded PDFs.

4. **View the Answer:**
   - The app retrieves relevant data and displays the generated answer along with the supporting text.

---

## **Technical Details**

### **Libraries Used**
- **Streamlit:** For creating the user interface.
- **LangChain:** For building the QA pipeline and integrating with Groq LLM.
- **Pinecone:** For hybrid dense-sparse retrieval of text.
- **HuggingFace Transformers:** For embedding generation.
- **PyPDFLoader:** For extracting text from PDFs.
- **BM25Encoder:** For sparse matrix representation.

### **Index Structure**
- The Pinecone index combines dense embeddings and sparse matrix representations, enabling hybrid search.
- The index dimension is set to 384 for compatibility with the chosen embedding model.

---

