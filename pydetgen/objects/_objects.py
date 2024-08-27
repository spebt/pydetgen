class ConfigBlock:
    def __init__(self, objname: str, datatype: str, data):
        self.objname = objname
        self.datatype = datatype
        self.data = data

    def __str__(self):
        output = self.objname + ": {"
        output += "dataType:" + self.datatype + "}"
        return output
