import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class addlayer():
    def __init__(self,his,bias,div,node):
        his.append(self.addinitialweight(div,node))
        bias.append(self.addinitialbias(node))

    def addinitialweight(self ,div, node):
        initialweight = np.random.rand(div,node)
        return initialweight/10
    def addinitialbias(self,node):
        initialbias = np.random.rand(node,1)
        return initialbias/10



class caculate():
    def splitx(self,x,y,his,bias,epoch):
        neuans=np.zeros((1,x.shape[1]))
        min=1
        saveerror=np.zeros((int(epoch/10)))
        for tes in range(epoch):
            errorrate=0
            for n in range(x.shape[1]):
                xx=np.array([[x[0][n]],[x[1][n]]])
                ans,taphis,taphisx=self.feedforword(xx,his,bias)
                neuans[0][n]=ans
                #print(taphisx[0].shape)
                #print(y[n])
                self.backpropagation(ans,y[n],his,bias,taphis,taphisx)
                errorrate+=abs(y[n]-ans)
            print(errorrate/4)
            if min > (errorrate/4):
                min = errorrate/4
            if tes%10 == 0 :
                saveerror[int(tes/10)]=errorrate
                pass
        t = np.arange(0.0, epoch/10, 1)
            
        print("min : " , min)
        #fig, ax = plt.subplots()
        #ax.plot(t, saveerror)
        #plt.show()
        
        return neuans

    def feedforword(self,x,his,bias):
        taphis=[]
        taphisx=[]
        tempx=x
        for i in range(len(his)):
            tempy=np.zeros((his[i].shape[1],1))
            for ii in range(his[i].shape[0]):
                for iii in range(his[i].shape[1]):
                    tempy[iii][0]=his[i][ii][iii]*tempx[ii][0]+tempy[iii][0]
            for c in range(len(tempy)):
                tempy[c][0]=tempy[c][0]-bias[i][c][0]
                
            taphis.append(tempy)
            for c in range(len(tempy)):
                #print(-tempy[c][0])
                if abs(tempy[c][0])>700:
                    tempy[c][0]=1*tempy[c][0]/abs(tempy[c][0])
                else:
                    tempy[c][0]=1/(1+m.exp(-tempy[c][0]))

            taphisx.append(tempx)

            tempx=tempy
        return tempy[0][0],taphis,taphisx

    def backpropagation(self,y,yd,his,bias,taphis,taphisx,learnrate=0.75):
        temperror=self.gather_error(y,yd,his,bias,taphis)
        for i in range(len(his)):
            c=len(his)-i-1
            for ii in range(his[c].shape[1]):
                bias[c][ii][0] -= learnrate*temperror[i][ii][0]*(1/(1+m.exp(-taphis[c][ii][0])))*(1-(1/(1+m.exp(-taphis[c][ii][0]))))
                for iii in range(his[c].shape[0]):
                    his[c][iii][ii] -= learnrate*temperror[i][ii][0]*(1/(1/(1+m.exp(-taphis[c][ii][0]))))*(1-(1/(1/(1+m.exp(-taphis[c][ii][0])))))*taphisx[c][iii][0]
                    pass
    
    def gather_error(self,y,yd,his,bias,taphis):
        temperror=[]
        errorinitial=np.zeros((1,1))
        errorinitial[0][0]=yd-y
        temperror.append(errorinitial)
        for i in range(len(his)-1):
            #print(temperror[i].shape)
            c=len(his)-i-1
            temp=np.zeros((his[c].shape[0],1))
            for ii in range(his[c].shape[1]):
                temp1=temperror[i][ii][0]
                for iii in range(his[c].shape[0]):
                    temp[iii][0] += temp1*(1/(1/(1+m.exp(-taphis[c][ii][0]))))*(1-(1/(1/(1+m.exp(-taphis[c][ii][0])))))*his[c][iii][ii]
            temperror.append(temp)
        return temperror


class caculate2():
    def __init__(self,x,y,his,bias,epoch):
        #self.feedforword(x,his)
        
        print(self.splitx(x,y,his,bias,epoch))
        pass

    def splitx(self,x,y,his,bias,epoch):
        neuans=np.zeros((1,x.shape[1]))
        min=1
        saveerror=np.zeros((int(epoch/10)))
        for tes in range(epoch):
            errorrate=0
            for n in range(x.shape[1]):
                xx=np.array([[x[0][n]],[x[1][n]]])
                ans,taphis,taphisx=self.feedforword(xx,his,bias)
                neuans[0][n]=ans
                #print(taphisx[0].shape)
                #print(y[n])
                self.backpropagation(ans,y[n],his,bias,taphis,taphisx)
                errorrate+=abs(y[n]-ans)
            print(errorrate/4)
            if min > (errorrate/4):
                min = errorrate/4
                minhis=his
            if tes%10 == 0 :
                saveerror[int(tes/10)]=errorrate
                pass
        t = np.arange(0.0, epoch/10, 1)
            
        print("min : " , min)
        #his=minhis
        #fig, ax = plt.subplots()
        #ax.plot(t, saveerror)
        #plt.show()
        
        return neuans

    def feedforword(self,x,his,bias):
        taphis=[]
        taphisx=[]
        tempx=x
        for i in range(len(his)):
            tempy=np.zeros((his[i].shape[1],1))
            for ii in range(his[i].shape[0]):
                for iii in range(his[i].shape[1]):
                    tempy[iii][0]=his[i][ii][iii]*tempx[ii][0]+tempy[iii][0]
            for c in range(len(tempy)):
                tempy[c][0]=tempy[c][0]-bias[i][c][0]
                
            taphis.append(tempy)
            for c in range(len(tempy)):
                #print(-tempy[c][0])
                if abs(tempy[c][0])>700:
                    tempy[c][0]=1*tempy[c][0]/abs(tempy[c][0])
                    #tempy[c][0]=1/(1+m.exp(-tempy[c][0]))
                else:
                    tempy[c][0]=1/(1+m.exp(-tempy[c][0]))

            taphisx.append(tempx)

            tempx=tempy
        return tempy[0][0],taphis,taphisx

    def backpropagation(self,y,yd,his,bias,taphis,taphisx,learnrate=0.75):
        temperror=self.gather_error(y,yd,his,bias,taphis)
        for i in range(len(his)):
            c=len(his)-i-1
            for ii in range(his[c].shape[1]):
                #temperror[i][ii][0]
                bias[c][ii][0] -= learnrate*temperror[i][ii][0]*(1/(1+m.exp(-taphis[c][ii][0])))*(1-(1/(1+m.exp(-taphis[c][ii][0]))))
                for iii in range(his[c].shape[0]):
                    his[c][iii][ii] -= learnrate*temperror[i][ii][0]*(1/(1/(1+m.exp(-taphis[c][ii][0]))))*(1-(1/(1/(1+m.exp(-taphis[c][ii][0])))))*taphisx[c][iii][0]
                    pass
    
    def gather_error(self,y,yd,his,bias,taphis):
        temperror=[]
        errorinitial=np.zeros((1,1))
        errorinitial[0][0]=yd-y
        temperror.append(errorinitial)
        for i in range(len(his)-1):
            #print(temperror[i].shape)
            c=len(his)-i-1
            temp=np.zeros((his[c].shape[0],1))
            for ii in range(his[c].shape[1]):
                temp1=temperror[i][ii][0]
                for iii in range(his[c].shape[0]):
                    temp[iii][0] += temp1*(1/(1/(1+m.exp(-taphis[c][ii][0]))))*(1-(1/(1/(1+m.exp(-taphis[c][ii][0])))))*his[c][iii][ii]
            
            temperror.append(temp)
        return temperror

class update():
    def __init__(self,x,y,yd,his,bias,taphis,learnrate=0.001):
        pass

    def backpropagation(self,y,yd,his,bias,taphis,taphisx,learnrate=0.001):
        for i in range(len(his)):
            c=len(his)-i-1
            for ii in range(his[c].shape[0]):
                for iii in range(his[c].shape[1]):
                    his[c][ii][iii] += learnrate*(yd-y)*(1/(1-taphis[c][ii]))*(1-(1/(1-taphis[c][ii])))*taphisx[c][iii]
                    pass
            pass
        
class train(caculate):
    def __init__(self,x,y,his,bias,epoch) -> None:
        super().__init__()
        self.splitx(x,y,his,bias,epoch)

class plot(caculate):
    def __init__(self,dimension,his,bias) -> None:
        super().__init__()
        x1,x2,z = self.createvariable(dimension,his,bias)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_zlim(0, 1.5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    


    def createvariable(self, dimension,his,bias):
        xx1 = np.arange(0, 1, 1/dimension)
        xx2 = np.arange(0, 1, 1/dimension)
        x1, x2 = np.meshgrid(xx1, xx2)
        z=np.zeros((dimension,dimension))
        for i in range(x1.shape[0]):
            for ii in range(x2.shape[0]):
                #print("in")
                xx=(np.array([[xx1[i]],[xx2[ii]]]))
                #print(xx.shape)
                
                t1,_,_=self.feedforword(xx,his,bias)
                z[i][ii]=t1
        return x1,x2,z



def main():
    his=[]
    bias=[]
    learnrate=0.0001
    x1=np.array([0,1,0,1])
    x2=np.array([0,1,1,0])
    x=np.array([x1,x2])
    y=np.array([0,0,1,1])
    addlayer(his,bias,2,3)
    addlayer(his,bias,3,3)
    addlayer(his,bias,3,1)
    #addlayer(his,bias,3,1)
    #xx=np.array([[x1[1]],[x2[1]]])
    #print(x.shape)
    #print(xx.shape)
    caculate2(x,y,his,bias,10000)
    #train(x,y,his,bias,10000)
    for i in bias:
        print(i.shape)
    
    plot(100,his,bias)
main()


#p=initial(30)
#caculate2(p)
#print(p.gg)

'''

aa=np.random.rand(1,4)
print(aa)
temp = []
temp.append(aa)
print(len(temp))
print(temp[0].shape)
'''