#Connecting Azure Blob

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import streamlit as st
from azure.storage.blob import BlobServiceClient
from azureml.core import Workspace, Dataset


# Establish a connection to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string("Yatish-azure_connection_string")

# Access the blob container
container_client = blob_service_client.get_container_client("Yatish-azure_container_name")

# Use Streamlit to display data
blobs_list = container_client.list_blobs()
for blob in blobs_list:
    st.write(blob.name)
   # Uploaded to Azure blob
    

#Further, I would typically use the data in my Azure Blob Storage as input for my machine learning models in Azure AI Studio. 
# This would involve uploading Yatish-azure data to Azure AI Studio and using it in Yatish-azure experiments, which is done through the Azure AI Studio interface or its SDK.

#Creating a connection for Azure AI studio

# Connect to Yatish-azure Azure Machine Learning workspace  
# The Workspace.from_config() reads the file config.json for details about Yatish-azure Azure Machine Learning workspace.
ws = Workspace.from_config()

# Upload Yatish-azure data from Azure Blob Storage to Azure AI Studio
datastore = ws.get_default_datastore()
datastore.upload(src_dir='Yatish-azure_local_directory', target_path='Yatish-azure_target_path')

# Register the uploaded data as a dataset in Azure AI Studio
dataset = Dataset.File.from_files(path=(datastore, 'Yatish-azure_target_path'))
dataset = dataset.register(workspace=ws, name='Yatish-azure_dataset_name')


#Credhub