from flask import jsonify
class Prediction:

    def __init__(self,result=None,score=0,b64=""):
        self._result=True if result>=0.8 else False
        self._score=round(score,3)
        self._b64=b64

    @property
    def result(self):
        return self._result
    
    @result.setter
    def result(self,value):
        self._result=value
    
    @result.deleter
    def result(self):
        del self._result

    
    @property
    def score(self):
        return self._score
    
    @score.setter
    def score(self,value):
        self._score=value

    @score.deleter
    def score(self):
        del self._score

    
    @property
    def b64(self):
        return self._b64
    
    @b64.setter
    def b64(self,value):
        self._b64=value
    
    @b64.deleter
    def b64(self):
        del self._b64

    def to_jsonify(self):
        return jsonify({
            "score":self.score,
            "result":self.result,
                        })