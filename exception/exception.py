import sys
import inspect
'''
    File to create the exception class 
'''

class ProjectError(Exception):

    def __init__(self, error_message, error_details: sys):
        
        super().__init__()

        self.error_message = error_message
        _, _, self.exception_traceback = error_details.exc_info()

        if self.exception_traceback is not None:
            self.file_name = self.exception_traceback.tb_frame.f_code.co_filename
            self.line_num = self.exception_traceback.tb_lineno
        else:
                        
            caller_frame = inspect.currentframe().f_back
            self.file_name = caller_frame.f_code.co_filename if caller_frame else "Unknown"
            self.line_num = caller_frame.f_lineno if caller_frame else 0

    def __str__(self):

        return "Error occured in file [{0}] line number [{1}] message [{2}]".format(self.file_name, self.line_num, str(self.error_message))    


