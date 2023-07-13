class value:
    def __init__(self,data,children=()):
        self.data=data
        self.grad=0
        #variables used for backward propagation 
        self._backward=lambda:None
        self._prev=set(children)
    
    def __add__(self,other):
        if(not isinstance(other,value)): other=value(other)
        parent=value(self.data+other.data,(self,other))
        
        def backward():
            self.grad+=parent.grad
            other.grad+=parent.grad
        parent._backward=backward                
        
        return parent
    
    def __mul__(self,other):
        other= other if(isinstance(other,value)) else value(other)
        
        parent=value(self.data*other.data,(self,other))
        def backward():
            self.grad+=other.data*parent.grad
            other.grad+=self.data*parent.grad
        parent._backward=backward                
        return parent
    
    def __pow__(self,other):
        if(isinstance(other,(int,float))):
            parent=value((self.data)**(other),(self,))

        def backward():
            self.grad=other*(self.data**(other-1))*parent.grad
        parent._backward=backward
        return parent
    
    def relu(self):
        parent=value(0 if self.data<=0 else self.data)

        def backward():
            self.grad=(0 if self.data<=0 else parent.grad)
        parent._backward=backward
        return parent
    
    def __neg__(self):
        return self*(-1)
    
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        return self+(-other)
    
    def __rsub__(self,other):
        return (-self)+other
    
    def __rmul__(self,other):
        return self * other
    
    def runbackpropagation(self):
        self.grad=1
        topo=[]    
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        for v in reversed(topo):
            v._backward()
        
    def __repr__(self):     #this function runs when print is called
        return f"data_object = {self.data}"