import logging
from datetime import datetime
import os

LOGS_FİLE_NAME=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".log"
logs_dir="logs"

os.makedirs(logs_dir,exist_ok=True)
file_name=os.path.join(logs_dir,LOGS_FİLE_NAME)

logging.basicConfig(filename=file_name,
                    level=logging.INFO,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

logger=logging.getLogger("logger")