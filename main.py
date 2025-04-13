from sentence_transformers import SentenceTransformer; # for sentence embedding
import faiss; # for vector similarity search
import numpy as np; # for numerical operations
from http import HTTPStatus; # for HTTP status codes
from openai import OpenAI; # for calling OpenAI/DeepSeek API

from dotenv import load_dotenv; # for loading environment variables
import os; # for environment variables
import sys; # for system operations

load_dotenv(); # load environment variables from .env file

# we don't need to tokenize in parallel
# to avoid running lots of models which would cause conflicts and deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false";

import warnings

# To use the warning
warnings.warn("This is a deprecation warning", PendingDeprecationWarning)


from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
)

def load_ducument(file_path):
    """
    This function is used to load the document from the local directory.
    :param file_path: the path of the document
    :return: the string content of the document
    """

    DOCUMENT_LOADER_MAPPING = {
        ".pdf": (PDFPlumberLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
    }

    ext = os.path.splitext(file_path)[1]
    if ext not in DOCUMENT_LOADER_MAPPING:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    loader_class, loader_args = DOCUMENT_LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    documents =  loader.load()
    content = "\n".join([doc.page_content for doc in documents])

    print(f"Loaded {len(documents)} documents from {file_path}, the head is: {content[:100]}")
    return content


def load_folder(folder_path):
    """
    This function is used to load the files in the folder.
    :param folder_path: the path of the folder
    :return: the string content of the files
    """
    all_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            document_text = load_ducument(file_path)
            print(f"Loaded {filename} successfully, the character length is {len(document_text)}.")

            # You can choose different text splitters to split the document.
            # For example, you can use RecursiveCharacterTextSplitter, SpacyTextSplitter, etc.
            text_splitter = SpacyTextSplitter( 
                chunk_size=512,
                chunk_overlap=128,
                pipeline="zh_core_web_sm", # for spaCy Chinese
            )
            chunks = text_splitter.split_text(document_text)
            print(f"The total number of chunks in the {filename} is {len(chunks)}.")
            all_chunks.extend(chunks)
        elif os.path.isdir(file_path):
            all_chunks.extend(load_folder(file_path))
    return all_chunks
    

def load_embedding_model():
    """
    This function is used to load the embedding model from the local directory.
    :return: the embedding model(bge-small-zh-v1.5)
    """
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(os.path.abspath("bge-small-zh-v1.5"))
    print("Embedding model loaded successfully.")
    print(f"The largest length of the bge-small-zh-v1.5 model is {embedding_model.max_seq_length}.")
    return embedding_model;

def indexing_process(folder_path, embedding_model):
    """
    This function is used to index the files in the folder.
    1. Load the files.
    2. Split the files into chunks.
    3. Embed the chunks.
    4. Save the chunks to the vector database[faiss].
    :param folder_path: the path of the folder
    :param embedding_model: the embedding model
    :return: faiss embedding vector index and segmented text block original content list
    """

    # Load the files in the folder.
    all_chunks = load_folder(folder_path)

    # Convert the chunks into embeddings.
    # normalize_embeddings represents to normalize the embeddings to unit length,
    # for calculating the cosine similarity.
    embeddings = []
    for chunk in all_chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)
    print("The embeddings have been converted successfully.")

    # Convert the embeddings into a numpy array.
    # Because the faiss index requires the embeddings to be a numpy array.
    embeddings_np = np.array(embeddings)

    # Get the dimension of the embeddings.
    demension = embeddings_np.shape[1]

    # Use cosine similarity to create a Faiss index.
    index = faiss.IndexFlatIP(demension)
    index.add(embeddings_np)
    print("The Faiss index has been created successfully.")

    return index, all_chunks


def retrieval_process(query, index, chunks, embedding_model, top_k=3):
    """
    Retrieval Process: transform the query into an embedding vector,
    then use the Faiss index to find the top k most similar chunks,
    and return the original content of the chunks.
    :param query: the query
    :param index: the Faiss index
    :param chunks: the segmented text block original content list
    :param embedding_model: the embedding model
    :param top_k: the number of chunks to return
    """

    # Transform the query into an embedding vector.
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)

    # Transform the query embedding vector into a numpy array.
    query_embedding_np = np.array([query_embedding])

    # Use the Faiss index to find the top k most similar chunks.
    # When use cosine similarity, the bigger score means the smaller distance.
    distances, indices = index.search(query_embedding_np, top_k)
    
    print(f"The query is: {query}")
    print(f"The top {top_k} most similar chunks are:")
    
    results = []
    for i in range(top_k):
        # Get the original content of the chunk.
        result_chunk = chunks[indices[0][i]]
        print(f"chunk {i}: {result_chunk}")

        # Get the distance of the chunk.
        result_distance = distances[0][i]
        print(f"distance: {result_distance}")
        
        # Save the result_chunk to the result list.
        results.append(result_chunk)
    
    print(f"Retrieval success!")
    return results
    
    
def generate_process(query, chunks):
    """
    Generate Process: invoke LLM API to generate the answer according to the query and the retrieved chunks.
    :param query: the query
    :param chunks: the segmented text block original content list
    """
    
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"reference document {i+1}: \n{chunk}\n\n"
    
    prompt = f"answer the following question based on the reference documents: \n{query}\n\nreference documents: \n{context}"
    
    print(f"The prompt is: {prompt}")
    
    # Invoke LLM API to generate the answer.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        generated_response = ""
        print("Start generating the answer...")
        
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                generated_response += content
                print(content, end="", flush=True)
                sys.stdout.flush()
        print("\nGenerating the answer completed!")
        return generated_response

    except Exception as e:
        print(f"Error invoking LLM API: {e}")
        return None
    
def main():
    print("Hello from rag-demo!")

    query = "下面报告中涉及了几个行业的案例以及总结各自面临的挑战？"
    embedding_model = load_embedding_model()
    index, chunks = indexing_process("fixtures", embedding_model)
    chunks = retrieval_process(query, index, chunks, embedding_model)
    answer = generate_process(query, chunks)
    print(f"The answer is:\n{answer}")


if __name__ == "__main__":
    main()
