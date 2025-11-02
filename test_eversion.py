import numpy
import math
import random
from numpy.polynomial import Polynomial as P
import fractions

def delta(x,y):
    if x==y:
        return 1
    else:
        return 0

def scale(alpha, v):
    result = []
    for i in range(len(v)):
        result.append(alpha*v[i])
    return result

#Determine if list of vectors generates a pointed cone
#Pointed means that the cone contains no nontrivial linear subspaces of the
#ambient vector space
#The algorithm is as follows: we look for an "extremal ray" of the cone.
#Extremal ray means that if v1 is a point in the ray r, and
#v2, v3 are points in the cone such that v2+v3=v1, then
#v2, v3 are also in r.  It is easy to show that a cone is pointed
#if and only if it has at least one extremal ray.
#Assume the cone is pointed.
#Given any coordinate in the ambient vector space, if S is the subset
#of the list of vectors with positive (resp. negative) values of this coordinate,
#then at least one of the vectors in the list must generate an extremal ray.
#If a vector v does generate an extremal ray, then we can quotient out by the
#extremal ray to get a pointed cone in the quotient space.
#Conversely, if the quotient cone is pointed, then the original cone
#is pointed provided that it doesn't contain the one-dimensional
#subspace spanned by v, which is equivalent in this case to containing
#positive scalar multiples of both v and -v.
#The algorithm works recursively based on this process.
#If the cone is pointed, it returns 1, along with (as a certificate) a vector in the ambient space
#whose dot product is positive with all nonzero vectors in the list.
#If it isn't pointed, it returns 0, along with a positive linear combination of the vectors that
#equals the zero vector.            
def isPointedCone(veclist):
    #print('length of veclist',len(veclist))
    ambientdimension = len(veclist[0])
    curlowest = len(veclist)+1
    curp = -1
    curlist = []
    if len(veclist)==1:
        if (veclist[0]==0).all():
            return [1,numpy.ones(ambientdimension)]
        else:
            return [1, veclist[0]]
    #To minimize number of recursive function calls, look for the coordinate with smallest possible nonzero number of pos or neg entries
    for i in range(ambientdimension):
        numneg = 0
        numpos = 0
        neglist = []
        poslist = []
        for j in range(len(veclist)):
            if veclist[j][i] < 0:
                numneg = numneg+1
                neglist.append(j)
            elif veclist[j][i] > 0:
                numpos = numpos+1
                poslist.append(j)
        if numneg > 0 and numneg < curlowest:
            curlowest = numneg
            curp = i
            curlist = neglist
        if numpos > 0 and numpos < curlowest:
            curlowest = numpos
            curp = i
            curlist = poslist
    #print('curlist',curlist)
    #If curp is still -1 then all of the vectors are zero, so cone is pointed
    if (curp == -1):
        return([1,numpy.ones(ambientdimension)])
    for j in curlist:
        curvec = veclist[j]
        pivot = curvec[curp]
        if pivot < 0:
            sgn = -1
        else:
            sgn = 1
        quotientvecs = []
        for k in range(len(veclist)):
            if k != j:
                #print(pivot,veclist[k],curvec)
                r = abs(pivot)*veclist[k]-sgn*veclist[k][curp]*curvec
                #print(r)
                #If r is the zero vector, then one of the vectors is either zero or a scalar multiple of the pivot vector
                if (r==0).all() and veclist[k][curp]*pivot < 0:
                        a = numpy.zeros(len(veclist))
                        a[k] = abs(pivot)
                        a[j] = abs(veclist[k][curp])
                        return [0,a]
                else:    
                    r = list(r)
                    del r[curp]
                    quotientvecs.append(numpy.array(r))
        #print('quotientvecs',quotientvecs)
        result = isPointedCone(quotientvecs)
        #print(result)
        #If we find that the quotient cone is not pointed, then ray j is not extremal, so try to find a linear dependence, or delete the ray if it's
        #a positive linear combination of the other vectors
        #If the quotient cone is pointed, use the result to find a certificate vector
        if (result[0]==0):
            m = result[1]
            l = list(m[0:j])
            l.append(0)
            l.extend(list(m[j:]))
            r = 0
            for k in range(len(veclist)):
                r = r+l[k]*veclist[k][curp]
            #print('r',r,'pivot',pivot,'l',l)
            if (r*pivot <= 0):
                dep = numpy.array(l)
                dep = abs(pivot)*dep+abs(r)*numpy.identity(len(veclist))[j,...]
                return [0,dep]
            else:
                vcopy = list(veclist)
                del vcopy[j]
                result = isPointedCone(vcopy)
                if (result[0]==1):
                    return result
                else:
                    m = result[1]
                    l = list(m[0:j])
                    l.append(0)
                    l.extend(list(m[j:]))
                    return [0,numpy.array(l)]
        else:
            m = result[1]
            #print(m,curp)
            l = list(m[0:curp])
            l.append(0)
            l.extend(list(m[curp:]))
            mlift = l
            #print(mlift, curvec)
            dotp = dot(mlift,curvec)
            mlift = scale(abs(pivot),mlift)
            mlift[curp] = -sgn*dotp
            #print('mlift',mlift)
            maxabs = 0
            for k in range(len(veclist)):
                if k != j:
                    d = dot(mlift,veclist[k])
                    if d!=0:
                        maxabs = max(1/d*abs(veclist[k][curp]),maxabs)
            #print('maxabs',maxabs)
            #print(('before: ',dot(mlift,curvec),mlift,curvec))
            if pivot > 0:
                mlift = plus(plus(mlift,scale(maxabs,mlift)),[fractions.Fraction(delta(curp,i),1) for i in range(ambientdimension)])
            else:
                mlift = minus(plus(mlift,scale(maxabs,mlift)),[fractions.Fraction(delta(curp,i),1) for i in range(ambientdimension)])
            #print(('after: ',dot(mlift,curvec),mlift,curvec))
            return [1,mlift]
        


def det(a,b,c,d,e,f,g,h,i):
    return a*e*i-a*f*h-(b*d*i-b*g*f)+(c*d*h-c*e*g)

def createFaceToEdge(faces):
    edgeD = dict()
    for x in faces:
        if frozenset([x[0],x[1]]) in edgeD:
            edgeD[frozenset([x[0],x[1]])].append(x)
        else:
            edgeD[frozenset([x[0],x[1]])] = [x]
        if frozenset([x[1],x[2]]) in edgeD:
            edgeD[frozenset([x[1],x[2]])].append(x)
        else:
            edgeD[frozenset([x[1],x[2]])] = [x]
        if frozenset([x[0],x[2]]) in edgeD:
            edgeD[frozenset([x[0],x[2]])].append(x)
        else:
            edgeD[frozenset([x[0],x[2]])] = [x]
    return edgeD

def createJoints(faceToEdge):
    joints = []
    for y in faceToEdge:
        x = faceToEdge[y]
        joints.append((set(x[0])).union(set(x[1])))
    return joints

def createEdgeToFace(faceToEdge):
    edge_to_face = dict()
    for y in faceToEdge:
        x = faceToEdge[y]
        edge_to_face[frozenset(set(x[0]).union(set(x[1])))] = (set(x[0])).intersection(set(x[1]))
    return edge_to_face

def createTriangleToVertex(faces):
    triangle_to_vertex = dict()
    for x in faces:
        for y in faces:
            x0 = set(x)
            y0 = set(y)
            if len(x0.intersection(y0))==1:
                intV = list(x0.intersection(y0))[0]
                if (frozenset(x),intV) not in triangle_to_vertex:
                    triangle_to_vertex[(frozenset(x),intV)] = set()
                for q in y:
                    if q != intV:
                        triangle_to_vertex[(frozenset(x),intV)].add(q)
    return triangle_to_vertex

def vertexToEdge(faces):
    vDict = dict()
    for x in faces:
        for y in x:
            if y not in vDict:
                vDict[y] = set()
            for z in x:
                if z != y:
                    vDict[y].add(z)
    return vDict
            

def secondOrderEdge(vertex_to_edge):
    sDict = dict()
    for x in vertex_to_edge:
        if x not in sDict:
            sDict[x] = set()
        for q in vertex_to_edge[x]:
            for y in vertex_to_edge[q]:
                if y!=x:
                    sDict[x].add(y)
    return sDict

def createFPairs(faces):
    fPairs = []
    for x in faces:
        for y in faces:
            x0 = set(x)
            y0 = set(y)
            if len(x0.intersection(y0))==1:
                fPairs.append([x,y])
    return fPairs

def testFractionalVectors(vlist):
    for x in vlist:
        for y in x:
            if type(y)!=fractions.Fraction:
                raise Exception('not a fraction')

def average(x,y):
    return [(x[0]+y[0])/2,(x[1]+y[1])/2,(x[2]+y[2])/2]

def dot(x,y):
    result = 0
    for i in range(len(x)):
        result += x[i]*y[i]
    return result

def minus(x,y):
    result = []
    for i in range(len(x)):
        result.append(x[i]-y[i])
    return result

def plus(x,y):
    result = []
    for i in range(len(x)):
        result.append(x[i]+y[i])
    return result

def qEval(p0,p1,p2,t):
    return p0+p1*t+p2*t*t

def cross(x,y):
    return [x[1]*y[2]-x[2]*y[1],-x[0]*y[2]+x[2]*y[0],x[0]*y[1]-x[1]*y[0]]

#orthogonal projection of v1 onto v2
def orthoProjection(v1,v2):
    res = minus(v1,scale(dot(v1,v2)/dot(v2,v2),v2))
    #print(dot(res,v2))
    return res

def randomFract():
    c = [random.gauss(0,2) for x in range(3)]
    return [fractions.Fraction(int(10000*x),10000) for x in c]

#Test if the quadratic p0+p1*x+p2*x^2 is strictly positive for t0 <= x <= t1
def testQuadraticPos(p0,p1,p2,t0,t1):
    #print((p0,p1,p2,t0,t1))
    ft = fractions.Fraction
    if type(p0)!=ft or type(p1)!=ft or type(p2)!=ft or type(t0)!=ft or type(t1)!=ft:
        raise Exception('arguments are not fractions')
    if t0 >= t1: 
        raise Exception('Test interval is empty or a point')
    if p2==0:
        if qEval(p0,p1,p2,t0) > 0 and qEval(p0,p1,p2,t1) > 0:
            return True
        else:
            return False
    vertex = -p1/(2*p2)
    if vertex <= t1 and vertex >= t0:
        if qEval(p0,p1,p2,vertex) > 0 and qEval(p0,p1,p2,t0) > 0 and qEval(p0,p1,p2,t1) > 0:
            return True
        else:
            return False
    else:
        if qEval(p0,p1,p2,t0) > 0 and qEval(p0,p1,p2,t1) > 0:
            return True
        else:
            return False

#x1,y1,z1,x2,y2,z2,r  rational 3-vectors
#adjacent faces sharing an edge
#rx(z1+(z2-z1)t).(x1+(x2-x1)t)
#Consider two triangles in 3-space that have the 
def separateJointRecursive(x1,y1,z1,x2,y2,z2,level):
    #Consider the 3x3 determinant as a function on R^9.
    #Given x1,y1,z2,x2,y2,z2 3-vectors, we can consider the convex hull of the 8 points
    #(x1,y1,z1),(x1,y1,z2),(x1,y2,z2),(x2,y2,z2),(
    #If the determinant is strictly positive or stricly negative on all 8 points,
    #then it follows by multilinearity that it is strictly positive (resp. negative)
    #on the whole convex hull.  
    d1 = det(x1[0],x1[1],x1[2],y1[0],y1[1],y1[2],z1[0],z1[1],z1[2])
    d2 = det(x2[0],x2[1],x2[2],y1[0],y1[1],y1[2],z1[0],z1[1],z1[2])
    d3 = det(x2[0],x2[1],x2[2],y2[0],y2[1],y2[2],z1[0],z1[1],z1[2])
    d4 = det(x2[0],x2[1],x2[2],y2[0],y2[1],y2[2],z2[0],z2[1],z2[2])
    d5 = det(x1[0],x1[1],x1[2],y2[0],y2[1],y2[2],z1[0],z1[1],z1[2])
    d6 = det(x2[0],x2[1],x2[2],y1[0],y1[1],y1[2],z2[0],z2[1],z2[2])
    d7 = det(x1[0],x1[1],x1[2],y2[0],y2[1],y2[2],z2[0],z2[1],z2[2])
    d8 = det(x1[0],x1[1],x1[2],y1[0],y1[1],y1[2],z2[0],z2[1],z2[2])
    if (d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0 and d8 > 0):
        return True
    if (d1 < 0 and d2 < 0 and d3 < 0 and d4 < 0 and d5 < 0 and d6 < 0 and d7 < 0 and d8 < 0):
        return True
    r = orthoProjection(average(average(x1,y1),average(x2,y2)),z1)
    #r = randomFract()
    testFractionalVectors([x1,y1,z1,x2,y2,z2,r])
    #print(level)
    if level > 20:
        #print('Could not separate triangle pair after subdividing 10 times')
        r = randomFract()
    if level > 30:
        return False
    poly1 = [dot(cross(r,z1),x1),dot(cross(r,minus(z2,z1)),x1)+dot(cross(r,z1),minus(x2,x1)),
             dot(cross(r,minus(z2,z1)),minus(x2,x1))]
    poly2 = [dot(cross(r,z1),y1),dot(cross(r,minus(z2,z1)),y1)+dot(cross(r,z1),minus(y2,y1)),
             dot(cross(r,minus(z2,z1)),minus(y2,y1))]
    if poly1[0] < 0:
        if (testQuadraticPos(-poly1[0],-poly1[1],-poly1[2],fractions.Fraction(0),fractions.Fraction(1)) and
                testQuadraticPos(poly2[0],poly2[1],poly2[2],fractions.Fraction(0),fractions.Fraction(1))):
            return True
        else:
            a1 = separateJointRecursive(x1,y1,z1,average(x1,x2),average(y1,y2),average(z1,z2),level+1)
            a2 = separateJointRecursive(average(x1,x2),average(y1,y2),average(z1,z2),x2,y2,z2,level+1)
            if a1==False or a2==False:
                return False
            else:
                return True
    elif poly1[0] > 0:
        if (testQuadraticPos(poly1[0],poly1[1],poly1[2],fractions.Fraction(0),fractions.Fraction(1)) and
                testQuadraticPos(-poly2[0],-poly2[1],-poly2[2],fractions.Fraction(0),fractions.Fraction(1))):
            return True
        else:
            a1 = separateJointRecursive(x1,y1,z1,average(x1,x2),average(y1,y2),average(z1,z2),level+1)
            a2 = separateJointRecursive(average(x1,x2),average(y1,y2),average(z1,z2),x2,y2,z2,level+1)
            if a1==False or a2==False:
                return False
            else:
                return True
    else:
        a1 = separateJointRecursive(x1,y1,z1,average(x1,x2),average(y1,y2),average(z1,z2),level+1)
        a2 = separateJointRecursive(average(x1,x2),average(y1,y2),average(z1,z2),x2,y2,z2,level+1)
        if a1==False or a2==False:
            return False
        else:
            return True

#x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,r rational 3-vectors
#faces sharing a vertex but not an edge    
#x1,y1,v1 and x2,y2,v2 are vertices of the first triangle at times 1 and 2
#z1,w1,v1 and z2,w2,v2 are vertices of the second triangle at times 1 and 2
#triangles share the vertex v which is at position v1 at time 1 and v2 at time 2
def testSeparatePoint(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,r):
    testFractionalVectors([x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,r])
    if dot(r,minus(x1,v1)) > 0:
        if dot(r,minus(y1,v1)) > 0 and dot(r,minus(x2,v2)) > 0 and dot(r,minus(y2,v2)) > 0 and dot(r,minus(z1,v1)) < 0 and dot(r,minus(w1,v1)) < 0 and dot(r,minus(z2,v2)) < 0 and dot(r,minus(w2,v2)) < 0:
            return True
        else:
            return False
    elif dot(r,minus(x1,v1)) < 0:
        if dot(r,minus(y1,v1)) < 0 and dot(r,minus(x2,v2)) < 0 and dot(r,minus(y2,v2)) < 0 and dot(r,minus(z1,v1)) > 0 and dot(r,minus(w1,v1)) > 0 and dot(r,minus(z2,v2)) > 0 and dot(r,minus(w2,v2)) > 0:
            return True
        else:
            return False
    else:
        return False

#x1,y1,z1,w1,x2,y2,z2,w2,v1,v2 rational 3-vectors
#faces sharing a vertex but not an edge
#x1,y1,v1 and x2,y2,v2 are vertices of the first triangle at times 1 and 2
#z1,w1,v1 and z2,w2,v2 are vertices of the second triangle at times 1 and 2
#triangles share the vertex v which is at position v1 at time 1 and v2 at time 2
def separateRecursive(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,level):
    testFractionalVectors([x1,y1,z1,w1,x2,y2,z2,w2,v1,v2])
    if level > 10:
        print('Could not separate triangle pair after subdividing 10 times')
        return False
    sp = isPointedCone([numpy.array(minus(v1,x1)),numpy.array(minus(v1,y1)),numpy.array(minus(z1,v1)),numpy.array(minus(w1,v1)),
                            numpy.array(minus(v2,x2)),numpy.array(minus(v2,y2)),numpy.array(minus(z2,v2)),numpy.array(minus(w2,v2))])
    if sp[0]==0:
            avx = average(x1,x2)
            avy = average(y1,y2)
            avz = average(z1,z2)
            avw = average(w1,w2)
            avv = average(v1,v2)
            res1 = separateRecursive(x1,y1,z1,w1,avx,avy,avz,avw,v1,avv,level+1)
            res2 = separateRecursive(avx,avy,avz,avw,x2,y2,z2,w2,avv,v2,level+1)
            if res1 and res2:
                return True
            else:
                return False
    else:
        if testSeparatePoint(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,sp[1])==True:
            return True
        else:
            print((x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,sp[1]))
            return False

def separatePair(fPairs,pt1,pt2):
    badCount = 0
    for p in fPairs:
        cVertex = set(p[0]).intersection(set(p[1]))
        cVertex = list(cVertex)[0]
        u = set(p[0]).union(set(p[1]))
        oV1 = [x for x in p[0] if x != cVertex]
        oV2 = [x for x in p[1] if x != cVertex]
        x1 = pt1[oV1[0]]
        y1 = pt1[oV1[1]]
        z1 = pt1[oV2[0]]
        w1 = pt1[oV2[1]]
        x2 = pt2[oV1[0]]
        y2 = pt2[oV1[1]]
        z2 = pt2[oV2[0]]
        w2 = pt2[oV2[1]]
        v1 = pt1[cVertex]
        v2 = pt2[cVertex]
        res = separateRecursive(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,0)
        if res:
            pass
        else:
            print(p)
            badCount += 1
    return badCount

def separateJoint(faceToEdge,pt1,pt2):
    badCount = 0
    for e in faceToEdge:
        fList = faceToEdge[e]
        f1 = fList[0]
        f2 = fList[1]
        edge = [x for x in e]
        allPts = set(f1).union(set(f2))
        oppPts = [x for x in allPts if x not in e]
        #r = average(x1 ,y1)
        x1 = minus(pt1[oppPts[0]],pt1[edge[0]])
        y1 = minus(pt1[oppPts[1]],pt1[edge[0]])
        #r = average(x1,y1)
        z1 = minus(pt1[edge[1]],pt1[edge[0]])
        x2 = minus(pt2[oppPts[0]],pt2[edge[0]])
        y2 = minus(pt2[oppPts[1]],pt2[edge[0]])
        z2 = minus(pt2[edge[1]],pt2[edge[0]])
        if separateJointRecursive(x1,y1,z1,x2,y2,z2,0)==False:
            print(fList)
            badCount += 1
    return badCount

def fr(v):
    return([fractions.Fraction(x) for x in v])

#Sanity checks that the methods for testing whether triangles intersect each other
#actually work in some simple cases
def runTests():
    print('Testing separateRecursive...')
    #Triangles that share one point at the origin.  At t=0, the triangles are the same.
    x1 = fr([1,1,0])
    y1 = fr([1,0,0])
    z1 = fr([1,1,1])
    w1 = fr([1,0,1])
    x2 = fr([1,1,0])
    y2 = fr([1,0,0])
    z2 = fr([1,1,0])
    w2 = fr([1,0,0])
    v1 = fr([0,0,0])
    v2 = fr([0,0,0])
    assert(separateRecursive(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,0)==False)
    z2 = fr([1,1,0.5])
    w2 = fr([1,0,0.5])
    #print((z2,w2))
    assert(separateRecursive(x1,y1,z1,w1,x2,y2,z2,w2,v1,v2,0)==True)
    z1 = fr([1,0,0])
    y1 = fr([1,1,0])
    x1 = fr([1,1,1])
    z2 = fr([1,0,0])
    y2 = fr([1,1,0])
    x2 = fr([1,1,0])
    r = fr([1,1,0.5])
    print('Testing separateJointRecursive...')
    assert(separateJointRecursive(x1,y1,z1,x2,y2,z2,0)==False)
    x2 = fr([1,1,-1])
    assert(separateJointRecursive(x1,y1,z1,x2,y2,z2,0)==False)
    x2 = fr([1,1,0.75])
    assert(separateJointRecursive(x1,y1,z1,x2,y2,z2,0)==True)
    print('All tests passed')

g = open('./faces.js','r')
fcs = g.read()
g.close()

t = eval(fcs)
faceToEdge = createFaceToEdge(t)

h = open('./positions.js','r')
pos = h.read()
h.close()

p = eval(pos)

np = []
for r in p:
    q1 = {}
    for x in r:
        q1[x] = [fractions.Fraction(int(100000*r[x][0]+0.5)),fractions.Fraction(int(100000*r[x][1]+0.5)),
                 fractions.Fraction(int(100000*r[x][2]+0.5))]
    np.append(q1)

#Validate that the triangulation is really a triangulation of a sphere
edges = dict()
for f in t:
    e = [None,None,None]
    e[0] = frozenset([f[0],f[1]])
    e[1] = frozenset([f[1],f[2]])
    e[2] = frozenset([f[2],f[0]])
    for edge in e:
        if edge in edges:
            edges[edge].append(f)
        else:
            edges[edge] = [f]

#Assert each edge is connected to exactly two faces
for e in edges:
    assert(len(edges[e])==2)

connected_set = set([0])
v2e = vertexToEdge(t)
callstack = [0]

while len(callstack) > 0:
    cv = callstack.pop()
    for x in v2e[cv]:
        if x not in connected_set:
            connected_set.add(x)
            callstack.append(x)
            
#Assert triangulation is of a connected object
assert(len(connected_set)==len(v2e))
#Assert Euler characteristic = 2
assert(len(t)-len(edges)+len(connected_set)==2)

runTests()
print("First check is for any edges where the two connected triangles cross over each other during some part of the linear movement in each stage")
for x in range(len(p)-1):
    print(str(separateJoint(faceToEdge,np[x],np[x+1]))+' bad edges found in stage '+str(x))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
fPairs = createFPairs(t)
print("Second check is for any vertices that have non-adjacent triangles intersecting each other during some part of the linear movement in each stage")
for x in range(len(p)-1):
    print(str(separatePair(fPairs,np[x],np[x+1]))+' bad vertices found in stage '+str(x))

