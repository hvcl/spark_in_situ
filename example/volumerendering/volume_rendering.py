from pyspark import SparkContext, SparkConf

import numpy
import sys
import getpass
from hdfs import InsecureClient
import time


#########################
# Set configuration
#########################
SPARK_PATH="/home/smhong/spark"
GPU_API_PATH= "/home/smhong/spark_hvcl/import/"
ImagePath = '/home/smhong/water_data'

flag_GPUcache = True
flag_GPUshuffle = True
 
dimx = 1000
dimy = 1000


#########################

import sys
sys.path.append(GPU_API_PATH)
from visAPIs import *


def init_patch(name,data,dim,halo=0,profiler=None):
   
    idx = int(name[name.rfind('_')+1:])

    data = numpy.fromstring(data,dtype=numpy.float32).reshape(tuple(dim)+(11,))
    
    newdata = numpy.zeros((dim[0]+2*halo,dim[1]+2*halo,dim[2]+2*halo,11),dtype=numpy.float32)
    
    newdata[halo:halo+dim[0],halo:halo+dim[1],halo:halo+dim[2],:] = data

    return (name,newdata) 

def get_data_range(name,data_shape,split_shape,halo=0):
   
    #halo = 0
    data = numpy.zeros(26).astype(numpy.int32) 

    idx = int(name[name.rfind('_')+1:])
    block_num = reduce(lambda x,y:x*y,split_shape)
    
    local_data_shape = map(lambda x,y:x/y,data_shape,split_shape)    
    
    start=[0,0,0]
    end=local_data_shape[::-1] 

    nx = split_shape[2]
    ny = split_shape[1]

    z = idx/(nx*ny)
    y = (idx/nx)%ny
    x = idx%nx

    start[0] += x*local_data_shape[2] 
    end  [0] += x*local_data_shape[2] 
    start[1] += y*local_data_shape[1] 
    end  [1] += y*local_data_shape[1] 
    start[2] += z*local_data_shape[0] 
    end  [2] += z*local_data_shape[0] 


    data[:3] = start
    data[4:7] = end
    data[8:11] = [0,0,0]
    data[12:15] = data_shape[::-1]
    data[16:19] = start
    data[20:23] = end
  
    data[0] -= halo
    data[1] -= halo
    data[2] -= halo
    data[4] += halo
    data[5] += halo
    data[6] += halo
    data[16] -= halo
    data[17] -= halo
    data[18] -= halo
    data[20] += halo
    data[21] += halo
    data[22] += halo


    data_shape = numpy.array(data_shape[::-1])

    data[:3]    -= data_shape/2
    data[4:7]   -= data_shape/2 
    data[8:11]  -= data_shape/2
    data[12:15] -= data_shape/2
    data[16:19] -= data_shape/2
    data[20:23] -= data_shape/2 
    #data[24] = halo
 
    return data

def get_z_idx(name,num,data_shape,split_shape,mmtx):
       
   # import random 
    #return  int(name[name.rfind('_')+1:])

    idx = int(name[name.rfind('_')+1:])

    local_data_shape = map(lambda x,y:x/y,data_shape,split_shape)    
    
    start=[0,0,0]
    end=local_data_shape[::-1] 

    nx = split_shape[2]
    ny = split_shape[1]

    z = idx/(nx*ny)
    y = (idx/nx)%ny
    x = idx%nx

    start[0] += x*local_data_shape[2] 
    end  [0] += x*local_data_shape[2] 
    start[1] += y*local_data_shape[1] 
    end  [1] += y*local_data_shape[1] 
    start[2] += z*local_data_shape[0] 
    end  [2] += z*local_data_shape[0] 
 

    def make_depth(mmtx,start,end):
        x,y,z,depth = 0, 0, 0, 0
        
        x = (start[0] + end[0])/2
        y = (start[1] + end[1])/2
        z = (start[2] + end[2])/2

        Z = mmtx[2][0]*x+mmtx[2][1]*y+mmtx[2][2]*z
        return Z

    return (make_depth(mmtx,start,end), start, end, name)


def get_first_key(name,block_list):

    num_image = len(block_list)
    idx = block_list.index(name)

    return idx


def getOnlyTag(data):
  
    t1 = time.time() 
    tags = []

    host_name = get_host_name()
 
    if isinstance(data,list) == False:
        tags.append([data[4:16],host_name])

    else :
        for elem in data:
            tags.append([elem[4:16],host_name])
    
    t2 = time.time() 

    #print t2-t1
    return tags




if __name__ == "__main__":


    ImgDim = [-1,-1,-1]
    ImgSplit = [1,1,1]

    halo = 1

    
    #client = InsecureClient('http://emerald:50070',user=username)
    #with client.read(ImageName +'/.meta', encoding='utf-8') as reader:
    #    content = reader.read().split('\n')
     
    meta = open(ImagePath + '/.meta').read().split('\n')

    for elem in meta:
        if elem.startswith('X : '):
            ImgDim[2] = int(elem[4:])
        if elem.startswith('Y : '):
            ImgDim[1] = int(elem[4:])
        if elem.startswith('Z : '):
            ImgDim[0] = int(elem[4:])
        if elem.startswith('X split : '):
            ImgSplit[2] = int(elem[10:])
        if elem.startswith('Y split : '):
            ImgSplit[1] = int(elem[10:])
        if elem.startswith('Z split : '):
            ImgSplit[0] = int(elem[10:])

    print ImgDim
    print ImgSplit


    LocalDim = map(lambda x,y:x/y,ImgDim,ImgSplit)    


    print "Total Size" , ImgDim
    print "Screen Size", dimx, dimy
    print "GPU cache", flag_GPUcache
    print "GPU shuffle",flag_GPUshuffle 
    

    xoff, yoff = 0, 0
    xoff = -dimx/2
    yoff = -dimy/2

    def getInfo(filename):
        data = numpy.fromstring(open(filename).read(),dtype=numpy.float32)
        mmtx = data[:16].reshape(4,4)
        inv  = data[16:32].reshape(4,4)
        ray_dir = data[32:35]
        transfer_func = data[35:]

        return mmtx,inv,ray_dir,transfer_func
 
    def getRotate(filename): 
        data = numpy.fromstring(open(filename).read(),dtype=numpy.float32)
        off = 16+16+3

        rotate_set =[]
        for i in range(10):
            mmtx = data[i*off:i*off+16].reshape(4,4)
            inv  = data[i*off+16:i*off+32].reshape(4,4)
            ray_dir = data[i*off+32:i*off+35]
            rotate_set.append([mmtx,inv,ray_dir])
  
        return rotate_set

    mmtx,inv,ray_dir,transfer_func = getInfo("info.txt")
    rotate_set = getRotate("rotate.txt")

    gpuinit(SPARK_PATH + "/conf/slaves")
    node_list = getnodelist(SPARK_PATH + "/conf/slaves")
    print node_list
    time.sleep(5)

    #sersock = open_port()
   
    t0 = time.time()    
 
    sc = SparkContext()

    sc.addFile(GPU_API_PATH + "/visAPIs.py")

    ImagePatch = sc.binaryFiles(ImagePath,1)

    ImagePatch = ImagePatch.map(lambda (name,data): init_patch(name,data,LocalDim,halo))

    ImagePatch = ImagePatch.map(lambda (name,data): (name,sendGPU(data)))
    
    if flag_GPUcache :
        ImagePatch = ImagePatch.map(lambda (name,data): (name,cacheGPU(data,True)))
            
    ImagePatch = ImagePatch.cache()

    ImageList = ImagePatch.map(lambda (name,data): (name)).collect()
    
    t1 = time.time()    
   
    print "Loading Time : ",t1-t0
 
    import math

    ptx_code = open('ray_casting.ptx','r').read()
    base_level = 4
    for i in range(10):
        print "Screen ",i
        mmtx, inv, ray_dir = rotate_set[i]
        try :

            t_list = []
            t_list.append(time.time())
            
            z_idx_list = map(lambda (name) : (get_z_idx(name,i,ImgDim,ImgSplit,mmtx)),ImageList)
        
            z_idx_list = sorted(z_idx_list,reverse = True)

            sorted_block_list= []

            for elem in z_idx_list:
                sorted_block_list.append(elem[3])

            imgs = ImagePatch.map(lambda (name,data): (get_first_key(name,sorted_block_list),execGPU(data,ptx_code,"render",[get_data_range(name,ImgDim,ImgSplit,halo),dimx,dimy,transfer_func,inv,xoff,yoff,11,ray_dir],dimx*dimy*4*4,[dimx,dimy])))


            if flag_GPUshuffle:
                imgs = imgs.map(lambda (name,data):(name,cacheGPU(data)))
            else :
                imgs = imgs.map(lambda (name,data):(name,recvGPU(data)))

            group_level = []
            block_num = len(sorted_block_list)
       
            cur_level = base_level 
            while cur_level <= block_num:
                group_level.append(base_level)
                cur_level *= base_level 

            sum_group_level = reduce(lambda x,y:x*y,group_level)

            if block_num/sum_group_level > 1:
                group_level.append(block_num/sum_group_level)

            #print group_level 

            tt0 = 0
            #for ll in range(len(group_level)): 
            #for ll in range(len(group_level)):
            cur_level = 0 
            for level in group_level:
                #print "Level %d"%level
                isFinal   = False
                if cur_level == len(group_level)-1:
                    isFinal = True
                cur_level += 1 

                def sort_image(data):
                    data = sorted(data)
    
                    arr = []

                    for elem in data:
                        arr.append(elem[1]) 

                    return arr               

    
                #imgs = imgs.map(lambda(idx,data): (idx/2,(idx,data))).reduceByKey(union)
                imgs = imgs.map(lambda(idx,data): (idx/level,(idx,data))).groupByKey().mapValues(list)
                imgs = imgs.map(lambda(idx,data): (idx,sort_image(data)))
                #imgs = imgs.map(lambda(idx,data): (idx/2,(idx,data))).groupByKey().mapValues(list)
        
 
                def arrange_after(after):
            
                    new_after =[]

                    for elem in after:
                        for data in elem:
                            new_after.append(data)
    
                    return sorted(new_after)


                if flag_GPUshuffle :
            
                    imgs = imgs.cache()

                    after = imgs.map(lambda(idx,data):(getOnlyTag(data))).collect()
   
                    #print "Original" ,after
 
                    after = arrange_after(after)
   
                    #print "Arrange", after
 
                    t_list.append(time.time())
     

                    tt1 = time.time()
                    newShuffle(after,node_list)
                    tt2 = time.time()

                    tt0 += tt2-tt1
                    t_list.append(time.time())
            

                #print cur_level, level
                func_args = [level,dimx,dimy]

                if flag_GPUshuffle:


                    imgs = imgs.map(lambda (name,data): (name,getSeq(data)))
                    if isFinal == True:
                        imgs = imgs.map(lambda (name,data): (name,execGPU(data,ptx_code,"composite_uchar",func_args,dimx*dimy*3,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,saveFile(data,"screen_%02d.raw"%i,dimx*dimy*3)))
                        imgs = imgs.map(lambda (name,data):(name,cacheGPU(data)))
                    else:
                        imgs = imgs.map(lambda (name,data): (name,execGPU(data,ptx_code,"composite",func_args,dimx*dimy*4*4,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,cacheGPU(data)))


                    #else :
                else :
                    imgs = imgs.map(lambda (name,data): (name,sendSeq(data)))
    
                    if isFinal == True:
                        imgs = imgs.map(lambda (name,data): (name,execGPU(data,ptx_code,"composite_uchar",func_args,dimx*dimy*3,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,saveFile(data,"screen_%02d.raw"%i,dimx*dimy*3)))
                        imgs = imgs.map(lambda (name,data):(name,cacheGPU(data)))
        
                    else : 
                        imgs = imgs.map(lambda (name,data): (name,execGPU(data,ptx_code,"composite",func_args,dimx*dimy*4*4,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,recv(data)))


 
            data = imgs.collect()

            t3 = time.time()
            t_list.append(time.time())
       
            print "\nTotal : ",t_list[-1] - t_list[0]
        except error as msg:
            print "Rendering Fail " + str(msg[0]) + " - " + str(msg[1])


    sc.stop()
    

