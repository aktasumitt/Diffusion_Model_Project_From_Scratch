from src.config.configuration import Configuration
from src.components.data_ingestion.data_ingestion import DataIngestion


class DataIngestionPipeline():
    def __init__(self):
        
        configuration = Configuration()
        self.dataingestionconfig = configuration.data_ingestion_config()

    def run_data_ingestion_pipeline(self):

        dataingestion = DataIngestion(self.dataingestionconfig)
        dataingestion.initiate_data_ingestion()


if __name__=="__main__":
    data_ingestion_pipeline=DataIngestionPipeline()
    data_ingestion_pipeline.run_data_ingestion_pipeline()