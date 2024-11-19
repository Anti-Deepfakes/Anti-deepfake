class SuccessResponse:
    def __init__(self, data):
        self.code = 200
        self.message = "요청에 성공하였습니다."
        self.data = data

    @staticmethod
    def ok(data):
        response = SuccessResponse(data)
        return response

    def dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }