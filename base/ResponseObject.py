import datetime
class ResponseObject():
    # final Date timestamp
    # int code;
    # String message;
    # T data;
    def __init__(self):
        super().__init__()
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def success(self, data, links = None, relationships = None):
        self.data = data
        self.message = "success"
        self.code = 200
        return self

    def error(self, code = 500, links = None, relationships = None, message= "error"):
        self.message = message
        self.code = code
        return self
