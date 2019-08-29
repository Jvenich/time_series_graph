
# coding: utf-8

# In[269]:

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import csv
import os

if __name__ == "__main__":
    
    N = 197
    mlc = 'Germany'
    fn_folder = '_' + mlc + '_' + 'new_data_nodes'
    try:
        os.mkdir(fn_folder)
    except:
        pass
    
    
#   find 'id' of the choosen country - mlc
    id, countries = get_country_id(mlc)

    save_nodes(N,mlc,countries)
        
#   P is the trade cost for all counter-parts of mlc over the whole years range
#   Y1, Y2 are the linear- and log- temporal derivatives respectively
    fn = fn_folder+'/'+ mlc+'_cost_1995-2015.png'
    fn2 =fn_folder+'/'+ mlc+'_distr_cost_1995-2015.png'

    P,Y1,Y2 = calculate_derivatives(N,id,1,fn,fn2)

#   calculate correlation matrix pij and the distances d, 
#   according to Eq. 1 in REF??    
#   following the paper I use only log-derivatives - Y2

#   (i) calculate p_ij and 
#   save edges for all the graph based on the entire time series    
    fn = fn_folder+'/'+ mlc+'_pij_d_1995-2015.png'
    fn2= fn_folder+'/'+ mlc+'_distrib_pij_d_1995-2015.png'

    
    pij,d = calculate_pij(N,Y2[1:21,:],1,fn, fn2)
    save_edges(N,mlc,fn_folder,0,countries,d,pij)

#   (ii) calculate p_ij and 
#   save edges for all the graph based on the 5-years moving window    

    for year in range(1995,2016-5):
        pij,d = calculate_pij(N,Y2[year-1995+1:year-1995+1+5,:])
        save_edges(N,mlc,fn_folder,year,countries,d,pij)


    


# In[268]:

def save_edges(N,mlc,fn_folder,year,countries,d,pij):
    if(year<1995):
        fn = fn_folder + '/' + mlc + '_all_average_pij_edges.csv'  
    else:
        fn = fn_folder + '/' + mlc + '_' + str(year) + '_' + str(year+5) + '_average_pij_edges.csv'  

    print(fn)

    with open(fn,'wt') as f:
        csv_writer = csv.writer(f)
        line = ['Source','Target','Weight','distance','pij','Countries','Year'] 
        csv_writer.writerow(line)
        print('done')
        
        for i in range(0,N):
            for j in range(0,i):
#                if(np.isnan(d[i][j]) == False):
                if( (d[i][j] >0) and (d[i][j]< np.sqrt(2)) ):
                    source = i
                    target = j
                    source_name = countries[i] 
                    target_name = countries[j]
                    name = [source_name + '-' +target_name]
                    w = 2 - d[i][j]
                    line = [source, target, w, d[i][j], pij[i][j], name, year] 
                    csv_writer.writerow(line)
   #     print('done')


# In[124]:

def save_nodes(N,mlc,countries):

    fn =  fn_folder + '/' + mlc +'_average_pij_nodes.csv'  
    print(fn)


    with open(fn,'wt') as f:
        csv_writer = csv.writer(f)
        line = ['Id','Label'] 
        csv_writer.writerow(line)
        
        for i in range(0,N): 
            line = [i, countries[i]] 
            csv_writer.writerow(line)
        


# In[259]:

def calculate_pij(N,Y, pic=0, fn=0, fnx=0):

    pij = np.zeros((N,N))

    for i in range(N):
        Yi = Y[:, i]
        indi = np.abs(Yi)<0.5
        for j in range(N):
            Yj = Y[:, j]
            indj = np.abs(Yj)<0.5

            ind = indi * indj
            
            term1 = np.mean( Yi[ind] * Yj[ind] )
            term2 = np.mean( Yi[ind] ) * np.mean( Yj[ind] )
        
            term3 = np.mean( Yi[ind]**2 ) - np.mean( Yi[ind] )**2
            term3 *=np.mean( Yj[ind]**2 ) - np.mean( Yj[ind] )**2
        
            pij[i][j] = (term1 - term2) / (np.sqrt(term3) + 1e-22)    

    d = np.sqrt(2*(1-pij))

    if(pic==1):
        fig = plt.figure(num=None, figsize=(10, 4), dpi=300, facecolor='w', edgecolor='k')
        plt.subplot(1,2,1)
        plt.pcolor(pij)
        plt.xlabel('Country id')
        plt.ylabel('Country id')
        plt.title('Correlation matrix, p')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolor(d)
        plt.title('Distance matrix, d')
        plt.ylabel('Country id')
        plt.xlabel('Country id')
        plt.colorbar()
        #plt.show()
        fig.savefig(fn)
        plt.close()
        
        fig2 = plt.figure(num=None, figsize=(10, 4), dpi=300, facecolor='w', edgecolor='k')
        plt.subplot(1,2,1)

        aa,bb = np.histogram(pij, bins=np.linspace(-1.1,1.1,100))
        center = (bb[:-1] + bb[1:]) / 2
        plt.plot(center,aa)
        aa,bb = np.histogram(d, bins=np.linspace(-0.1,2.1,100))
        center = (bb[:-1] + bb[1:]) / 2
        plt.plot(center,aa)
        plt.yscale('log')
        plt.ylabel('Number of counts')
        plt.xlabel('Value')


        plt.legend(('p','d'))
    
        plt.subplot(1,2,2)
        aa,bb = np.histogram(2-d, bins=np.linspace(-0.1,2.1,100))
        center = (bb[:-1] + bb[1:]) / 2
        plt.plot(center,aa)
        aa,bb = np.histogram(1/d, bins=np.linspace(-0.1,2.1,100))
        center = (bb[:-1] + bb[1:]) / 2
        plt.plot(center,aa)
        plt.yscale('log')
        plt.ylabel('Number of counts')
        plt.xlabel('Weight')


        plt.legend(('w=2-d','w=1/d'))

        fig2.savefig(fnx)

        plt.close()

        

    print('pij done')
    
    return pij, d



# In[236]:

def calculate_derivatives(N,id,pic=0,fn=0,fnx=0):
    
    P = np.zeros((2016-1995,N))
    dP = np.zeros((2016-1995,N)) # = Y
    dlnP = np.zeros((2016-1995,N)) # = Y in a log-scale

    o=0

    for year in range(1995,2016):
        fn2 = 'data_by_years/'+str(year)+'_20180521_ESCAP_WB_tradecosts_dataset.csv'
    
        x2  = pd.read_csv(fn2)    
        tmp = np.array(x2)
        data2 = tmp[id,:]

        P[o][:] = data2
    
        o=o+1
    
    tmax = o

    P = P + 1
        
    for t in range(1,tmax):
        dP[t][:] = P[t][:] - P[t-1][:]
        dlnP[t][:] = np.log(P[t][:]) - np.log(P[t-1][:])

        
    if(pic==1):
        tmp,count = get_country_id(mlc)
        ids = [38, 49, 64, 181]
        fig = plt.figure(num=None, figsize=(10, 4), dpi=300, facecolor='w', edgecolor='k')
        plt.subplot(1,2,1)
        xx = np.linspace(1995,2015,21);
        for i in range(4):
            plt.plot(xx,P[:,ids[i]])
        plt.xlim(1995,2015)
        plt.ylim(0,150)
        plt.title('Trade costs - Germany')
        plt.xlabel('Year')
        plt.ylabel('Trade cost [units]')
        plt.legend((count[ids[0]], count[ids[1]], count[ids[2]], count[ids[3]]))

 
        plt.subplot(1,2,2)
        xx = np.linspace(1996,2015,20);
        for i in range(4):
            plt.plot(xx,dlnP[1:21,ids[i]])
        plt.xlim(1995,2015)
        #plt.ylim(0,150)
        plt.title('Trade costs changing rate - Germany')
        plt.xlabel('Year')
        plt.ylabel('dLn( Trade cost [units/year] )')
 
        fig.savefig(fn)
        plt.close()
        
        fig2 = plt.figure(num=None, figsize=(20, 8), dpi=300, facecolor='w', edgecolor='k')
        plt.subplot(1,3,1)
        yrs = np.zeros(21)
        cm = plt.cm.get_cmap('RdYlBu_r',21)
        for o in range(0,21):
            tmp = P[o,:]-1
            aa,bb = np.histogram(tmp[tmp>0], bins=np.linspace(0,1000,30))
            center = (bb[:-1] + bb[1:]) / 2
            plt.plot(center,aa+2*o,color=cm(o))
            yrs[o] = (1995+o)
        plt.ylabel('Number of counts')
        plt.title('Trade costs values (arbitrary shifted)')


        plt.subplot(1,3,2)

        for o in range(1,21):
#            plt.hist(dlnP[o,:], normed=True, bins=20,histtype=u'step',color=cm(o))
            tmp = dlnP[o,:]
            aa,bb = np.histogram(tmp, bins=np.linspace(-8,8,30))
            center = (bb[:-1] + bb[1:]) / 2
            plt.plot(center,aa,color=cm(o))
            yrs[o] = (1996+o)
        plt.yscale('log')
        plt.title('Trade costs changing rate (dln)')


        plt.subplot(1,3,3)

        for o in range(1,21):
#            plt.hist(dlnP[o,:], normed=True, bins=20,histtype=u'step',color=cm(o))
            tmp = dlnP[o,:]
            aa,bb = np.histogram(tmp, bins=np.linspace(-0.5,0.5,30))
            center = (bb[:-1] + bb[1:]) / 2
            plt.plot(center,aa+2*o*0,color=cm(o))
            yrs[o] = (1996+o)
        plt.legend((yrs))

        plt.yscale('log')
        plt.title('Trade costs changing rate (dln)')
    
        fig2.savefig(fnx)
        plt.close()
        
    
    print('Derivatives done')
    
    return P, dP, dlnP
    


# In[127]:

def get_country_id(mlc):
    
    fn = 'data_by_years/1995_20180521_ESCAP_WB_tradecosts_dataset.csv'
    
    with open(fn,'r') as f:
        reader = csv.reader(f)
        countries = next(reader)
        
    id = countries.index(mlc)

    print(mlc,id)
    
    return id, countries


# In[5]:


    
    
    








# In[ ]:



