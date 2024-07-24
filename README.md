# PDF Content Analyzer

PDF Content Analyzer is a web application that extracts, processes, and analyzes content from PDF files. It uses advanced natural language processing (NLP) techniques to summarize, embed, and index document chunks, facilitating efficient retrieval and query answering. This project integrates Streamlit for the frontend, Milvus for vector storage and search, and OpenAI's GPT for summarization and query answering.

## Features

- **PDF Content Extraction**: Extracts text from PDF files.
- **Text Chunking**: Splits extracted text into manageable chunks.
- **Embedding and Indexing**: Creates embeddings for text chunks and stores them in Milvus.
- **Hierarchical Clustering and Summarization**: Uses Gaussian Mixture Models (GMM) and GPT to create hierarchical summaries.
- **Efficient Retrieval**: Utilizes Milvus for fast and scalable vector search.
- **Document Ranking**: Re-ranks retrieved documents using a cross-encoder for relevance.
- **Contextual Query Answering**: Constructs context from top-ranked documents and uses GPT to answer queries.


### Prerequisites

- Python 3.8 or higher
- Docker
- OpenAI API key

### REFRENCES
-**https://milvus.io/docs/integrate_with_sentencetransformers.md
-**https://platform.openai.com/usage
-**https://milvus.io/docs/install_standalone-docker.md

The project will take some time after uploading the files as the vecor embeddings takes time.
##IMPORTANT
Pls set you API KEY in .env file
Set your own port number and host id to connect milvus to docker
