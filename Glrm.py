import cv2
import numpy as np
class DegreeGLRLM:
    def __init__(self, mat_0, mat_45, mat_90, mat_135):
        self.Mat_0 = mat_0 
        self.Mat_45 = mat_45
        self.Mat_90 = mat_90
        self.Mat_135 = mat_135
        self.Degrees = [mat_0, mat_45, mat_90, mat_135]

class FeatureGLRLM:
    def __init__(self, sre, lre, glu, rlu, rpc):
        self.SRE = sre
        self.LRE = lre
        self.GLU = glu
        self.RLU = rlu
        self.RPC = rpc
        self.Features = [sre, lre, glu, rlu, rpc]
class Operator:
    def __init__(self):
        self.__degree_obj:DegreeGLRLM = None

    def __SRE(self):
        input_matrix = self.__degree_obj.Degrees
        matSRE = []
        for input_matrix in input_matrix:
            S = 0
            SRE = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]
                SRE += (Rj/S)/((x+1)**2)
            SRE = round(SRE, 3)
            matSRE.append(SRE)
        return round(sum(matSRE),3)
    
    def __LRE(self):
        input_matrix = self.__degree_obj.Degrees
        matLRE = []
        for input_matrix in input_matrix:
            S = 0
            LRE = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]
            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]
                LRE += (Rj * ((x + 1) ** 2)) / S
            LRE = round(LRE, 3)
            matLRE.append(LRE)
        return round(sum(matLRE),3)

    def __GLU(self):
        input_matrix = self.__degree_obj.Degrees
        matGLU = []
        for input_matrix in input_matrix:
            S = 0
            GLU = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]
            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]
                GLU += ((x + 1) ** 2) / S
            GLU = round(GLU, 3)
            matGLU.append(GLU)
        return round(sum(matGLU),3)

    def __RLU(self):
        input_matrix = self.__degree_obj.Degrees
        matRLU = []
        for input_matrix in input_matrix:
            S = 0
            RLU = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]
            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]
                RLU += (Rj ** 2) / S
            RLU = round(RLU, 3)
            matRLU.append(RLU)
        return round(sum(matRLU),3)

    def __RPC(self):
        input_matrix = self.__degree_obj.Degrees
        matRPC = []
        for input_matrix in input_matrix:
            S = 0
            RPC = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]
            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                RPC += (Rj) / (input_matrix.shape[0]*input_matrix.shape[1])
                # print('( ', (Rj), ' ) /', input_matrix.shape[0]*input_matrix.shape[1])
            RPC = round(RPC, 3)
            matRPC.append(RPC)
        # print('Perhitungan RPC')
        return round(sum(matRPC),3)
    
    def create_feature(self, degree:DegreeGLRLM):
        self.__degree_obj = degree
        return FeatureGLRLM(
            self.__SRE(), 
            self.__LRE(), 
            self.__GLU(), 
            self.__RLU(), 
            self.__RPC())

class Degree:
    def __init__(self):
        self.__image_src = None
        self.__gray_level = None
        self.__run_length = None
        
    def __degree0GLRLM(self):
        degree0Matrix = np.zeros([self.__gray_level, self.__run_length])
        counter = 0
        for y in range(self.__image_src.shape[0]):
            for x in range(self.__image_src.shape[1]):
                nowVal = self.__image_src[y][x]
                if x + 1 >= self.__image_src.shape[1]:
                    nextVal = None
                else:
                    nextVal = self.__image_src[y][x + 1]
                if nextVal != nowVal and counter == 0:
                    degree0Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree0Matrix[int(nowVal)][counter] += 1
                    counter = 0
        return degree0Matrix

    def __degree90GLRLM(self):
        degree90Matrix = np.zeros([self.__gray_level, self.__run_length])
        counter = 0
        for x in range(self.__image_src.shape[1]):
            for y in range(self.__image_src.shape[0]):
                nowVal = self.__image_src[y][x]
                if y + 1 >= self.__image_src.shape[0]:
                    nextVal = None
                else:
                    nextVal = self.__image_src[y + 1][x]
                if nextVal != nowVal and counter == 0:
                    degree90Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree90Matrix[int(nowVal)][counter] += 1
                    counter = 0
        return degree90Matrix

    def __degree45GLRLM(self):
        degree45Matrix = np.zeros([self.__gray_level, self.__run_length])
        for y in range(self.__image_src.shape[0]):
            counter = 0
            i_range = max(self.__image_src.shape)
            for i in range(i_range):
                y1 = y - i
                if i >= self.__image_src.shape[1] or y1 < 0:
                    break
                else:
                    nowVal = self.__image_src[y1][i]
                if y1 - 1 < 0 or i + 1 >= self.__image_src.shape[1]:
                    nextVal = None
                else:
                    nextVal = self.__image_src[y1 - 1][i + 1]
                if nextVal != nowVal and counter == 0:
                    degree45Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree45Matrix[int(nowVal)][counter] += 1
                    counter = 0
        for x in range(self.__image_src.shape[1]):
            if x == self.__image_src.shape[1] - 1:
                break
            counter = 0
            i_range = max(self.__image_src.shape)
            for i in range(i_range):
                y_i = -1 - i
                x_i = -1 + i - x
                # print(y_i, x_i)
                if x_i >= 0 or y_i <= -1 - self.__image_src.shape[0]:
                    break
                else:
                    nowVal = self.__image_src[y_i][x_i]

                if y_i - 1 <= -(self.__image_src.shape[0] + 1) or x_i + 1 >= 0:
                    nextVal = None
                else:
                    nextVal = self.__image_src[y_i - 1][x_i + 1]
                if nextVal != nowVal and counter == 0:
                    degree45Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree45Matrix[int(nowVal)][counter] += 1
                    counter = 0
        return degree45Matrix

    def __degree135GLRLM(self):
        degree135Matrix = np.zeros([self.__gray_level, self.__run_length])

        for y in range(self.__image_src.shape[0]):
            counter = 0
            i_range = max(self.__image_src.shape)
            for i in range(i_range):
                y1 = y + i
                if y1 >= self.__image_src.shape[0] or i >= self.__image_src.shape[1]:
                    break
                else:
                    nowVal = self.__image_src[y1][i]
                    if y1 >= self.__image_src.shape[0] - 1 or i >= self.__image_src.shape[1] - 1:
                        nextVal = None
                    else:
                        nextVal = self.__image_src[y1 + 1][i + 1]
                    if nextVal != nowVal and counter == 0:
                        degree135Matrix[int(nowVal)][counter] += 1
                    elif nextVal == nowVal:
                        counter += 1
                    elif nextVal != nowVal and counter != 0:
                        degree135Matrix[int(nowVal)][counter] += 1
                        counter = 0
        for x in range(self.__image_src.shape[1]):
            if x == 0:
                continue
            i_range = max(self.__image_src.shape)
            counter = 0
            for i in range(i_range):
                x1 = x + i
                if i >= self.__image_src.shape[0] or x1 >= self.__image_src.shape[1]:
                    break
                else:
                    nowVal = self.__image_src[i][x1]
                if i >= self.__image_src.shape[0] - 1 or x1 >= self.__image_src.shape[1] - 1:
                    nextVal = None
                else:
                    nextVal = self.__image_src[i + 1][x1 + 1]
                if nextVal != nowVal and counter == 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                    counter = 0
        return degree135Matrix

    
    def create_matrix(self, normalized_img, level):
        self.__image_src = normalized_img
        self.__gray_level = level
        self.__run_length = max(normalized_img.shape)
        
        mat0 = self.__degree0GLRLM()
        mat45 = self.__degree45GLRLM()
        mat90 = self.__degree90GLRLM()
        mat135 = self.__degree135GLRLM()
        
        return DegreeGLRLM(mat0, mat45, mat90, mat135)

class GLRLM:
    def __init__(self):
        self.__degree = Degree()
        self.__operator = Operator()
        super()
        
    
    
    def __normalization(self, value, min=0, max=255, newmin=0, newmax=10):
        newValue = (value-min)*((newmax-1-newmin)/(max-min))-newmin
        return newValue

    def normalizationImage(self, image=None, minScale=0, maxScale=9):
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                image[y][x] = round(
                    self.__normalization(value=image[y][x],
                                    newmin=minScale,
                                    newmax=maxScale,
                                    ))
        return image
        
    def __check_and_convert_to_gray(self, image):
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
        
        
    def get_features(self, image, level=8):
        grayscale_image = self.__check_and_convert_to_gray(image)
        normalized_image = self.normalizationImage(grayscale_image, 0, level)
        degree_obj = self.__degree.create_matrix(normalized_image, level)
        feature_obj = self.__operator.create_feature(degree_obj)
        return feature_obj

