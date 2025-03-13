import sys
from src.logger import logger


class ExceptionNetwork(Exception):
    
    def __init__(self,error:str,error_detail:sys):
        super(Exception,self).__init__()
        
        self.error=error
        _,_,error_message=error_detail.exc_info()
        
        line_no=error_message.tb_lineno
        file_name=error_message.tb_frame.f_code.co_filename
        
        self.ERROR_LOG=f"Error is detached file name: [{file_name} ], line: [{line_no}], Error: [{self.error}]"
        
        logger.error(self.ERROR_LOG)
        
    def __str__(self):
        return self.ERROR_LOG
        
        
        
if __name__=="__main__":
    
    try:
        a=10/0
        print(a)
    except Exception as e:
        raise ExceptionNetwork(e,sys)
    
    
