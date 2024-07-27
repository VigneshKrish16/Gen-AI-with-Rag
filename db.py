from neo4j import GraphDatabase
import streamlit as st

# Retrieve credentials from Streamlit secrets
neo4j_uri = st.secrets["NEO4J_URI"]
neo4j_user = st.secrets["NEO4J_USER"]
neo4j_password = st.secrets["NEO4J_PASSWORD"]

# Initialize Neo4j driver
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def fetch_documents():
    documents = []
    with driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN d.query AS query, d.content AS content")
        for record in result:
            documents.append({"query": record["query"], "content": record["content"]})
    return documents

documents = fetch_documents()

# Display documents in Streamlit
st.title("Document Queries and Contents")
for doc in documents:
    st.subheader(f"Query: {doc['query']}")
    st.write(f"Content: {doc['content']}")
    st.write("-----------")

driver.close()
