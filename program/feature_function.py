class FeatureFunction(object):
    def __init__(self, previous_label, label):
        self.previous_label = previous_label
        self.label = label
        self.weight = float(0)
        self.count = 0

    def apply(self, previous_label, label):
        return 1 if self.previous_label == previous_label and self.label == label else 0
    def apply_match(self,previous_label):
        return 1 if self.previous_label == previous_label else 0
    def update(self,n,w):
        self.weight = w
        self.count = n
    
    def print_value(self):
        print(self.previous_label,self.label,self.weight)
    
    def print_value1(self):
        print(self.previous_label,self.label)
    
    def get_weight(self):
        return self.weight
    
    def get_count(self):
        return self.count
    
    def put_weight(self,w):
        self.weight = w
        
    def print_value_file(self):
        value = str(self.previous_label)+" "+str(self.label)+" "+str(self.weight)
        f=open("features_example.txt","a+")
        f.write(value)
        f.write("\n")
        f.close()
    def __hash__(self):
        return hash(self.previous_label) ^ hash(self.label)

    def __eq__(self, other):
        return (self.previous_label, self.label) == (other.previous_label, other.label)
