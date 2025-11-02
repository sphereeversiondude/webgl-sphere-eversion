import numpy
import math
import torch

#Barycentric subdivision of a triangulation
def barycentric(faces, pos_list):
    edgeD = dict()
    cIdx = max([x for x in pos_list[0]])+1
    nFaces = []
    for x in faces:
        if frozenset([x[0],x[1]]) not in edgeD:
            edgeD[frozenset([x[0],x[1]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[0]]+y[x[1]])
            cIdx += 1
        if frozenset([x[1],x[2]]) not in edgeD:
            edgeD[frozenset([x[1],x[2]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[1]]+y[x[2]])
            cIdx += 1
        if frozenset([x[0],x[2]]) not in edgeD:
            edgeD[frozenset([x[0],x[2]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[0]]+y[x[2]])
            cIdx += 1
        nFaces.append([x[0],edgeD[frozenset([x[0],x[1]])],edgeD[frozenset([x[0],x[2]])]])
        nFaces.append([x[1],edgeD[frozenset([x[1],x[2]])],edgeD[frozenset([x[0],x[1]])]])
        nFaces.append([x[2],edgeD[frozenset([x[0],x[2]])],edgeD[frozenset([x[1],x[2]])]])
        nFaces.append([edgeD[frozenset([x[0],x[1]])],edgeD[frozenset([x[0],x[2]])],edgeD[frozenset([x[1],x[2]])]])
    return nFaces

#Modifies the triangulation at the very top and bottom of the sphere
#so we don't get dozens of triangles meeting at one vertex, which causes
#odd behavior when "smoothing" the eversion using gradient descent.
#Basic idea /|\  -> |\/|
def mutate(faces, pos_list, topNode, joints, frac, edge_to_face, interp_type):
    nFaces = []
    nPos = []
    even_joints = set()
    startJoint = None
    for x in joints:
        if topNode in x and topNode in edge_to_face[frozenset(x)]:
            startJoint = x
    evenJoints = set()
    cJoint = startJoint
    evenJoints.add(frozenset(startJoint))
    #We are cutting the number of edges in half with each iteration
    #Need to tweak it a little bit if the number of edges is odd
    isOdd = False
    print(startJoint)
    while cJoint != None:
        tJoint = cJoint
        cJoint = None
        for y in joints:
            if topNode in y and topNode in edge_to_face[frozenset(y)]:
                if len(y.intersection(tJoint)) == 2 and frozenset(y) not in evenJoints and len(y.intersection(startJoint)) != 3:
                    evenJoints.add(frozenset(y))
                    tJoint = y
                    cJoint = y
        if cJoint==None:
            for y in joints:
                if topNode in y and topNode in edge_to_face[frozenset(y)]:
                    if len(y.intersection(startJoint)) == 3 and len(y.intersection(tJoint))==2:
                        print(y)
                        isOdd = True
                        #This is a leftover "odd" triangle in the case we have an even number of edges
                        #Need to deal with it to make sure sphere is complete
                        center = edge_to_face[frozenset(y)]
                        link = [c for c in center if c != topNode]
                        last = [c for c in y if c not in y.intersection(startJoint)]
                        extraVertices = [last[0],link[0]]
                        print(extraVertices)
    print(tJoint)
    nvert = dict()
    cIdx = max([x for x in pos_list[0]])+1
    sf = 1
    for y in evenJoints:
        center = edge_to_face[y]
        sides = [x for x in y if x not in center]
        print(sides)
        opp = [c for c in center if c != topNode]
        print(opp,center,y)
        for x in sides:
            if x in nvert:
                pass
            else:
                for pos in pos_list:
                    scaleFact = 1
                    rScaleFact = 1
                    nVec = (1-frac)*pos[topNode]+frac*pos[x]
                    if pos != topNode:
                        scaleFact = numpy.linalg.norm(pos[x][0:2])/math.sqrt(3*3-pos[x][2]*pos[x][2])
                        sf = scaleFact
                        rScaleFact = scaleFact*math.sqrt(3*3-nVec[2]*nVec[2])/numpy.linalg.norm(nVec[0:2])
                    nVec = numpy.array([rScaleFact*nVec[0],rScaleFact*nVec[1],nVec[2]])
                    nvert[x] = (nVec,cIdx)
                    pos[cIdx] = nvert[x][0]
                interp_type[cIdx] = interp_type[topNode]
                cIdx += 1
        nFaces.append([sides[1],opp[0],nvert[sides[1]][1]])
        nFaces.append([nvert[sides[0]][1],opp[0],nvert[sides[1]][1]])
        nFaces.append([nvert[sides[0]][1],opp[0],sides[0]])
        nFaces.append([topNode,nvert[sides[0]][1],nvert[sides[1]][1]])
    for x in nvert:
        print(x)
    if isOdd==True:
        nFaces.append([nvert[extraVertices[0]][1],extraVertices[0],extraVertices[1]])
        nFaces.append([nvert[extraVertices[0]][1],nvert[extraVertices[1]][1],extraVertices[1]])
        nFaces.append([nvert[extraVertices[0]][1],nvert[extraVertices[1]][1],topNode])
    for x in faces:
        if topNode not in x:
            nFaces.append(x)
    return nFaces

#Construct the geometry in the middle of the sphere that
#is used in the "twisting" phase
def getMiddle(numLoops,insideProfile,outsideProfile,cintProfile,coutProfile,startIdx,topZ,bottomZ,r,gap):
    pts = dict()
    faces = []
    counter = 0
    ptTrip = []
    bottom_perim = []
    top_perim = []
    #Create first loop
    intR = insideProfile[0]
    outR = outsideProfile[0]
    cintR = cintProfile[0]
    coutR = coutProfile[0]
    outAdjust1 = []
    outAdjust2 = []
    inAdjust1 = []
    inAdjust2 = []
    leveldict = {}
    pairs = []
    pts[counter] = numpy.array([0,0,topZ+gap])
    counter += 1
    pts[counter] = numpy.array([0,0,bottomZ-gap])
    counter += 1
    leveldict[0] = len(insideProfile)-1
    leveldict[1] = 0
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([r*math.cos((2*y+0.75)*math.pi/numLoops),r*math.sin((2*y+0.75)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.0)*math.pi/numLoops),r*math.sin((2*y+1.0)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.25)*math.pi/numLoops),r*math.sin((2*y+1.25)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.75)*math.pi/numLoops),r*math.sin((2*y+1.75)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.0)*math.pi/numLoops),r*math.sin((2*y+2.0)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.25)*math.pi/numLoops),r*math.sin((2*y+2.25)*math.pi/numLoops),bottomZ])
        counter += 1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops),1])
        for i in range(-6,0):
            leveldict[counter+i] = 0
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+0.4)*math.pi/numLoops),outR[0]*math.sin((2*y+0.4)*math.pi/numLoops),outR[1]])
        counter += 1
        pts[counter] = numpy.array([coutR[0]*math.cos((2*y+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*y+1.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+1.6)*math.pi/numLoops),outR[0]*math.sin((2*y+1.6)*math.pi/numLoops),outR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+1.4)*math.pi/numLoops),intR[0]*math.sin((2*y+1.4)*math.pi/numLoops),intR[1]])
        counter += 1
        pts[counter] = numpy.array([cintR[0]*math.cos((2*y+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*y+2.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+2.6)*math.pi/numLoops),intR[0]*math.sin((2*y+2.6)*math.pi/numLoops),intR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
        for i in range(-6,0):
            leveldict[counter+i] = 0 
    for y in range(1,len(insideProfile)-1):
        baseCounter = counter
        for z in range(numLoops):
            outR = outsideProfile[y]
            intR = insideProfile[y]
            cintR = cintProfile[y]
            coutR = coutProfile[y]
            pts[counter] = numpy.array([outR[0]*math.cos((2*z+0.4)*math.pi/numLoops),outR[0]*math.sin((2*z+0.4)*math.pi/numLoops),outR[1]])
            counter += 1
            pts[counter] = numpy.array([coutR[0]*math.cos((2*z+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*z+1.0)*math.pi/numLoops),coutR[1]])
            counter += 1
            pts[counter] = numpy.array([outR[0]*math.cos((2*z+1.6)*math.pi/numLoops),outR[0]*math.sin((2*z+1.6)*math.pi/numLoops),outR[1]])
            counter += 1
            pairs.append([counter-1,counter-2,counter-3])
            pts[counter] = numpy.array([intR[0]*math.cos((2*z+1.4)*math.pi/numLoops),intR[0]*math.sin((2*z+1.4)*math.pi/numLoops),intR[1]])
            counter += 1
            pts[counter] = numpy.array([cintR[0]*math.cos((2*z+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*z+2.0)*math.pi/numLoops),cintR[1]])
            counter += 1
            pts[counter] = numpy.array([intR[0]*math.cos((2*z+2.6)*math.pi/numLoops),intR[0]*math.sin((2*z+2.6)*math.pi/numLoops),intR[1]])
            counter += 1
            pairs.append([counter-1,counter-2,counter-3])
            for x in range(6):
                faces.append([baseCounter+6*z+x,baseCounter+(6*z+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*z+x-6*numLoops])
                faces.append([baseCounter+6*z+x,baseCounter+(6*z+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*z+x+1)%(6*numLoops)])
            for i in range(-6,0):
                leveldict[counter+i] = y 
    intR = insideProfile[-1]
    outR = outsideProfile[-1]
    cintR = cintProfile[-1]
    coutR = coutProfile[-1]
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+0.4)*math.pi/numLoops),outR[0]*math.sin((2*y+0.4)*math.pi/numLoops),outR[1]])
        counter += 1
        pts[counter] = numpy.array([coutR[0]*math.cos((2*y+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*y+1.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+1.6)*math.pi/numLoops),outR[0]*math.sin((2*y+1.6)*math.pi/numLoops),outR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+1.4)*math.pi/numLoops),intR[0]*math.sin((2*y+1.4)*math.pi/numLoops),intR[1]])
        counter += 1
        pts[counter] = numpy.array([cintR[0]*math.cos((2*y+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*y+2.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+2.6)*math.pi/numLoops),intR[0]*math.sin((2*y+2.6)*math.pi/numLoops),intR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        for i in range(-6,0):
                leveldict[counter+i] = len(insideProfile)-1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([r*math.cos((2*y+0.75)*math.pi/numLoops),r*math.sin((2*y+0.75)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.0)*math.pi/numLoops),r*math.sin((2*y+1.0)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.25)*math.pi/numLoops),r*math.sin((2*y+1.25)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.75)*math.pi/numLoops),r*math.sin((2*y+1.75)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.0)*math.pi/numLoops),r*math.sin((2*y+2.0)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.25)*math.pi/numLoops),r*math.sin((2*y+2.25)*math.pi/numLoops),topZ])
        counter += 1
        for i in range(-6,0):
                leveldict[counter+i] = len(insideProfile)-1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops),0])
    return (faces,bottom_perim,top_perim,pts,leveldict,pairs)

#Functions used to look up faces corresponding to edges
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

#Linear interpolation
def interpolate(pos1,pos2,t):
    pos3 = dict()
    for x in pos1:
        pos3[x] = (1-t)*pos1[x]+t*pos2[x]
    return pos3

#"Twist" interpolation used during main phase of
#eversion.  In the middle of the sphere, given the initial and final points, x1, x2,
#the intermediate point will travel on the helix centered on
#the center of the sphere and midpoint of x1 and x2 that intersects both points.
#At the top and bottom of the sphere, the helix is centered on the midpoint of x1 and x2
#and is oriented straight up and down (instead of passing thru center of sphere).
def getTwistMapping(pos1,pos2,t,interp_type):
    pos3 = dict()
    offset = numpy.array([0,0,0])
    for x in pos1:
        if interp_type[x]==1:
            start = pos1[x]-offset;
            end = pos2[x]-offset;
            midpoint = (start+end)/2.;
            nmidpoint = midpoint/numpy.linalg.norm(midpoint)
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+offset;
        elif interp_type[x]==2:
            start = pos1[x];
            end = pos2[x];
            midpoint = (start+end)/2.;
            start = pos1[x]-midpoint;
            end = pos2[x]-midpoint;
            nmidpoint = numpy.array([0,0,-1])
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+midpoint;
        elif interp_type[x]==3:
            start = pos1[x]-offset;
            end = pos2[x]-offset;
            midpoint = (start+end)/2.;
            start = pos1[x]-midpoint;
            end = pos2[x]-midpoint;
            nmidpoint = numpy.array([0,0,1])
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+midpoint;
    return pos3

def round_floats(o):
    if isinstance(o, float): return round(o, 5)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

#Reorder vertices in each face to guarantee cross product points outward from center of sphere
def reorder_faces(faces,positions):
    for f in range(len(faces)):
        if numpy.dot(numpy.cross(positions[0][faces[f][2]]-positions[0][faces[f][0]],positions[0][faces[f][1]]-positions[0][faces[f][0]]),positions[0][faces[f][0]]) < 0:
            faces[f] = [faces[f][1],faces[f][0],faces[f][2]]

                                                                                            
if __name__ == '__main__':
    p1 = [[1,-2],[2.5,0.5],[2.7,0],[2.5,-0.5],[1,2]]
    p2 = [[0.5,-2],[2.5,1.5],[4.0,0],[2.5,-1.5],[0.5,2]]
    p3 = [[1,-2],[2.5,0.4],[2.5,0],[2.5,-0.4],[1,2]]
    p4 = [[0.5,-2],[2.5,1.8],[4.5,0],[2.5,-1.8],[0.5,2]]
    numLoops = 10
    mf = getMiddle(numLoops,p2,p1,p4,p3,0,2.5,-2.5,0.5,0.5)

    pairs = mf[5]
    levd = mf[4]
    oldPos = mf[3]
    interp_type = dict()

    newPos = dict()

    for x in pairs:
        if levd[x[0]]==2:
            newPos[x[0]] = oldPos[x[2]]
            newPos[x[2]] = oldPos[x[0]]
            newPos[x[1]] = oldPos[x[1]]
            interp_type[x[0]] = 1
            interp_type[x[1]] = 1
            interp_type[x[2]] = 1
        elif levd[x[0]]==1 or levd[x[0]]==3:
            newPos[x[0]] = numpy.array([oldPos[x[2]][0],oldPos[x[2]][1],-oldPos[x[2]][2]])
            newPos[x[2]] = numpy.array([oldPos[x[0]][0],oldPos[x[0]][1],-oldPos[x[0]][2]])
            newPos[x[1]] = numpy.array([oldPos[x[1]][0],oldPos[x[1]][1],-oldPos[x[1]][2]])
            interp_type[x[0]] = 1
            interp_type[x[1]] = 1
            interp_type[x[2]] = 1
        elif levd[x[0]]==0 or levd[x[0]]==4:
            if numpy.linalg.norm(oldPos[x[0]][0:2]) > 0.75:
                midpt = (oldPos[x[0]][0:2]+oldPos[x[2]][0:2])/2
                subtr = numpy.array([midpt[0],midpt[1],0])
                newPos[x[0]] = oldPos[x[2]]-0.5*subtr
                newPos[x[2]] = oldPos[x[0]]-0.5*subtr
                newPos[x[1]] = oldPos[x[1]]-0.5*subtr
            else:
                newPos[x[0]] = numpy.array([2*oldPos[x[2]][0],2*oldPos[x[2]][1],oldPos[x[2]][2]])
                newPos[x[2]] = numpy.array([2*oldPos[x[0]][0],2*oldPos[x[0]][1],oldPos[x[0]][2]])
                newPos[x[1]] = numpy.array([2*oldPos[x[1]][0],2*oldPos[x[1]][1],oldPos[x[1]][2]])
            if levd[x[0]] == 0:
                interp_type[x[0]] = 3
                interp_type[x[1]] = 3
                interp_type[x[2]] = 3
            else:
                interp_type[x[0]] = 2
                interp_type[x[1]] = 2
                interp_type[x[2]] = 2

    for x in oldPos:
        if x not in newPos:
            newPos[x] = numpy.array([-oldPos[x][0], -oldPos[x][1], oldPos[x][2]])
            if oldPos[x][2] < 0:
                interp_type[x] = 3
            else:
                interp_type[x] = 2

    final_inverted = dict()
    sinv = math.sin(math.pi/4/numLoops)
    cosv = math.cos(math.pi/4/numLoops)
    for x in pairs:
        avg = -(newPos[x[0]]+newPos[x[2]])/2
        tLength = math.sqrt(3*3-avg[2]*avg[2])
        avg2 = tLength*avg[0:2]/numpy.linalg.norm(avg[0:2])
        nAvg = numpy.array([cosv*avg2[0]+sinv*avg2[1],cosv*avg2[1]-sinv*avg2[0],-avg[2]])
        nAvg2 = numpy.array([cosv*avg2[0]-sinv*avg2[1],cosv*avg2[1]+sinv*avg2[0],-avg[2]])
        final_inverted[x[2]] = nAvg
        final_inverted[x[0]] = nAvg2
        final_inverted[x[1]] = numpy.array([avg2[0],avg2[1],-avg[2]])
    for x in newPos:
        if x not in final_inverted and x!=0 and x!=1:
            tLength = math.sqrt(3*3-newPos[x][2]*newPos[x][2])
            avg2 = tLength*newPos[x][0:2]/numpy.linalg.norm(newPos[x][0:2])
            final_inverted[x] = numpy.array([avg2[0],avg2[1],newPos[x][2]])
        elif x==0 or x==1:
            #final_inverted[x] = newPos[x]
            if x==0:
                final_inverted[x] = numpy.array([0.,0.,3.0])
            else:
                final_inverted[x] = numpy.array([0.,0.,-3.0])

    initial = dict()
    for x in final_inverted:
        initial[x] = -1*final_inverted[x]

    nf = mf[0]
    face_to_edge = createFaceToEdge(nf)
    edge_to_face = createEdgeToFace(face_to_edge)
    joints = createJoints(face_to_edge)
    pos_list = [initial,final_inverted,newPos,oldPos]
    joints = createJoints(face_to_edge)
    for x in range(3):
        nf = mutate(nf, pos_list, 0, joints, 0.8, edge_to_face, interp_type)
        face_to_edge = createFaceToEdge(nf)
        edge_to_face = createEdgeToFace(face_to_edge)
        joints = createJoints(face_to_edge)

    for x in range(3):
        nf = mutate(nf, pos_list, 1, joints, 0.8, edge_to_face, interp_type)
        face_to_edge = createFaceToEdge(nf)
        edge_to_face = createEdgeToFace(face_to_edge)
        joints = createJoints(face_to_edge)

    for x in initial:
        if initial[x][2]==2.5:
            print(x,initial[x])

    for x in final_inverted:
        if x==0 or x==1:
            pass
        else:
            cv = final_inverted[x]
            scaleFact = math.sqrt(3*3-cv[2]*cv[2])/numpy.linalg.norm(cv[0:2])
            nVec = numpy.array([scaleFact*cv[0],scaleFact*cv[1],cv[2]])
            final_inverted[x] = nVec

    for x in initial:
        if x==0 or x==1:
            pass
        else:
            cv = initial[x]
            scaleFact = math.sqrt(3*3-cv[2]*cv[2])/numpy.linalg.norm(cv[0:2])
            nVec = numpy.array([scaleFact*cv[0],scaleFact*cv[1],cv[2]])
            initial[x] = nVec
    

    twistArr = [final_inverted]
    for x in range(1,10):
        nx = interpolate(final_inverted,newPos,x/10)
        twistArr.append(nx)
    twistArr.append(newPos)
    for x in range(1,10):
        nx = getTwistMapping(newPos,oldPos,x/10,interp_type)
        twistArr.append(nx)
    twistArr.append(oldPos)
    for x in range(1,10):
        nx = interpolate(oldPos,initial,x/10)
        twistArr.append(nx)
    twistArr.append(initial)
    centerPts = set()

    for x in twistArr:
        z0 = barycentric(nf,[x])
        qf = barycentric(z0,[x])
    nf = qf
    #nf = z0
    face_to_edge = createFaceToEdge(nf)
    edge_to_face = createEdgeToFace(face_to_edge)
    joints = createJoints(face_to_edge)
    
    #Make the very top and bottom more spherical
    for x in range(len(twistArr)):
        for y in twistArr[x]:
            if abs(twistArr[x][y][2]) >= 2.5 and y!=0 and y!=1:
                scaleFact = numpy.linalg.norm(twistArr[x][2][0:2])/math.sqrt(3*3-twistArr[x][2][2]*twistArr[x][2][2])
                rScaleFact = scaleFact*math.sqrt(3*3-twistArr[x][y][2]*twistArr[x][y][2])/numpy.linalg.norm(twistArr[x][y][0:2])
                twistArr[x][y] = numpy.array([rScaleFact*twistArr[x][y][0],rScaleFact*twistArr[x][y][1],twistArr[x][y][2]])
            elif (x==0 or x==len(twistArr)-1) and y!=0 and y!=1:
                scaleFact = numpy.linalg.norm(twistArr[x][2][0:2])/math.sqrt(3*3-twistArr[x][2][2]*twistArr[x][2][2])
                rScaleFact = scaleFact*math.sqrt(3*3-twistArr[x][y][2]*twistArr[x][y][2])/numpy.linalg.norm(twistArr[x][y][0:2])
                twistArr[x][y] = numpy.array([rScaleFact*twistArr[x][y][0],rScaleFact*twistArr[x][y][1],twistArr[x][y][2]])

    for x in range(1,10):
        twistArr[x] = interpolate(twistArr[0],twistArr[10],x/10)

    for x in range(1,10):
        twistArr[20+x] = interpolate(twistArr[20],twistArr[30],x/10)

    twistNumpy = numpy.zeros((len(twistArr),len(twistArr[0]),3))
    
    for y in range(len(twistArr)):
        for x in twistArr[y]:
            for i in range(3):
                twistNumpy[y][x][i] = twistArr[y][x][i]

    
    cArr = torch.tensor(twistNumpy,requires_grad=True)

    mult_tensor = torch.ones_like(cArr)

    for x in range(cArr.shape[0]):
            if x==0 or x==cArr.shape[0]-1:
                for y in range(cArr.shape[1]):
                    for z in range(cArr.shape[2]):
                        mult_tensor[x][y][z] = 0.

    reorder_faces(nf,pos_list)

    indexSelect0 = []
    indexSelect1 = []
    indexSelect2 = []

    fcindex = dict()
    for f in enumerate(nf):
        fcindex[frozenset(f[1])] = f[0]

    edgeSelect0 = []
    edgeSelect1 = []

    edges0 = []
    edges1 = []
    for x in edge_to_face:
        e = list(edge_to_face[x])
        edges0.append(e[0]) 
        edges1.append(e[1])

    edges0 = torch.tensor(edges0)
    edges1 = torch.tensor(edges1)

    for x in edge_to_face:
        e = list(edge_to_face[x])
        opp_pts = [p for p in x if p not in e]
        edgeSelect0.append(fcindex[frozenset([e[0],e[1],opp_pts[0]])])
        edgeSelect1.append(fcindex[frozenset([e[0],e[1],opp_pts[1]])])
    
    for f in nf:
        indexSelect0.append(f[0])
        indexSelect1.append(f[1])
        indexSelect2.append(f[2])

    indexSelect0 = torch.tensor(indexSelect0)
    indexSelect1 = torch.tensor(indexSelect1)
    indexSelect2 = torch.tensor(indexSelect2)

    edgeSelect0 = torch.tensor(edgeSelect0)
    edgeSelect1 = torch.tensor(edgeSelect1)

    #The eversion is built now but in the middle of the sphere it's made of a bunch of
    #blocky polygon shapes.  To make it look smoother,
    #we will use gradient descent/pytorch
    for i in range(10000):
        print(i)
        pts0 = torch.index_select(cArr,1,indexSelect0)
        pts1 = torch.index_select(cArr,1,indexSelect1)
        pts2 = torch.index_select(cArr,1,indexSelect2)

        v1 = pts1-pts0
        v2 = pts2-pts0

        normals = torch.linalg.cross(v1,v2,dim=2)

        norms0 = torch.index_select(normals,1,edgeSelect0)
        norms1 = torch.index_select(normals,1,edgeSelect1)

        nn0 = torch.linalg.norm(norms0,dim=2)
        nn1 = torch.linalg.norm(norms1,dim=2)

        nnorms0 = norms0/nn0.unsqueeze(2)
        nnorms1 = norms1/nn1.unsqueeze(2)

        vq = nnorms0-nnorms1
        #This is a cost function that measures the distance between surface normals of adjacent faces.
        #If the faces are oriented nearly the same way (which we want to encourage)
        #then this will be very small
        tot = torch.linalg.norm(vq,dim=2)
        tot1 = tot*tot

        stage_length = tot1.shape[0]
        timeSelect0 = torch.tensor(range(0,stage_length-1))
        timeSelect1 = torch.tensor(range(1,stage_length))

        tnorms0 = torch.index_select(cArr,0,timeSelect0)
        tnorms1 = torch.index_select(cArr,0,timeSelect1)

        vq = tnorms0-tnorms1
        #This cost function measures the total of how much each point moves in each time step
        tot = torch.linalg.norm(vq,dim=2)
        tot2 = tot*tot

        
        #This cost function measures how much the edges deviate from their normal length in the original sphere position
        e0 = torch.index_select(cArr,1,edges0)
        e1 = torch.index_select(cArr,1,edges1)
        lengths = e0-e1
        flengths = torch.linalg.norm(lengths,dim=2)
        dlength = flengths-flengths[0].expand(flengths.shape[0],-1)
        dlength = dlength*dlength

        #Sum all the cost functions and do gradient descent
        fsum = torch.sum(tot1)+torch.sum(tot2)+torch.sum(dlength)
        #print((torch.sum(tot1),0.05*torch.sum(tot2),0.5*torch.sum(dlength)))
        fsum.backward()

        cArr.data = cArr.data-0.000005*mult_tensor*cArr.grad.data
        cArr.grad.data.zero_()
    
    twistArr = cArr
    posList = []
    for x in twistArr:
        nd = dict()
        for y in range(len(x)):
            nd[y] = x[y]
        posList.append(nd)
    
    temp_str = 'var coordDicts = $pos1; var faces = $faces;'
    temp_str = temp_str.replace('$faces',str(nf))
    
    pos_list = []
    for y in posList:
        subd=dict()
        for x in range(len(y)):
            subd[x] = list(numpy.copy(y[x].detach()))
            subd[x] = [float(str(y)) for y in subd[x]]
        pos_list.append(subd)
    pos_list = round_floats(pos_list)
    temp_str = temp_str.replace('$pos1',str(pos_list))

    g = open('./dataList.js','w')
    g.write(temp_str)
    g.close()

    g = open('./positions.js','w')
    g.write(str(pos_list))
    g.close()

    g = open('./faces.js','w')
    g.write(str(nf))
    g.close()
