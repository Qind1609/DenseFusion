import torch.utils.data as data
from PIL import Image
import os, sys
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import json
from scipy.spatial.transform import Rotation as R

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = 'datasets/ycb_nvidia_fat/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb_nvidia_fat/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        self.single = []
        self.mixed = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:7] == 'single/':
                self.single.append(input_line)
            else:
                self.mixed.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_single = len(self.single)
        self.len_mixed = len(self.mixed)

        class_file = open('datasets/ycb_nvidia_fat/dataset_config/classes.txt')
        
        # load objects pointcloud (store by class name)
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld['{0}'.format(class_input[:-1])] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld['{0}'.format(class_input[:-1])].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld['{0}'.format(class_input[:-1])] = np.array(self.cld['{0}'.format(class_input[:-1])])
            input_file.close()
    
        self.list_of_class = list(self.cld.keys())
        
        #need to resize (WxH)
        self.xmap = np.array([[j for i in range(960)] for j in range(540)])
        self.ymap = np.array([[i for i in range(960)] for j in range(540)])
        self.resize = transforms.Resize((540,960), interpolation=Image.NEAREST)
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        
        self.minimum_num_pt = 200
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
                                            self.trancolor,
                                            self.resize,
                                            ])

        self.symmetry_obj_idx = [12, 15, 18, 19, 20] #idx in class.txt
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.refine = refine
        
        # number object in 1 image
        self.front_num = 2

        print(len(self.list))

    def __getitem__(self, index):
         
        # get a RGB image
        img = Image.open('{0}/{1}.jpg'.format(self.root, self.list[index]))
        # get depth map
        depth = Image.open('{0}/{1}.depth.png'.format(self.root, self.list[index]))
        # get label (Segmentation image)
        label = Image.open('{0}/{1}.seg.png'.format(self.root, self.list[index]))
        meta = json.load(open('{0}/{1}.json'.format(self.root, self.list[index])))
        cam_setting = json.load(open('{0}/{1}_camera_settings.json'.format(self.root, self.list[index][:self.list[index].rfind('/')+1])))
        # get data from mix or single
        obj_setting = json.load(open('{0}/{1}_object_settings.json'.format(self.root, self.list[index][:self.list[index].rfind('/')+1])))
        
        camera = cam_setting['camera_settings'][0]['intrinsic_settings']
        cam_cx = camera['cx']
        cam_cy = camera['cy']
        cam_fx = camera['fx']
        cam_fy = camera['fy']

        img = self.resize(img)
        label = np.array(self.resize(label))
        depth = np.array(self.resize(depth))

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0)) #pixels which are background

        add_front = False
        
        obj = np.unique(label).tolist()[1:] #  seg_class_id of objects in the label image
        
        np.set_printoptions(threshold=sys.maxsize)
        #print('{0}/{1}.jpg'.format(self.root, self.list[index])) 
        #create noise label
        if self.add_noise:
            img = self.trancolor(img)
            for k in range(5):
                seed = random.choice(self.mixed)
                front = np.array(self.trancolor(self.resize(Image.open('{0}/{1}.jpg'.format(self.root, seed))).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))                    # change image from layer in depth to layer in stack
                f_label = np.array(self.resize(Image.open('{0}/{1}.seg.png'.format(self.root, seed))))
                front_label = np.unique(f_label).tolist()[1:] #remove element '0' -> array of seg_class_id 
                if len(front_label) < self.front_num: #get image which have number of objects >= (3)
                   continue
                front_label = random.sample(front_label, self.front_num) #choice random 2 object in image
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i)) #get mask of pixel which not object in each loop
                    if f_i == front_label[0]:
                        mask_front = mk             #   mask for the first chosen object in arr front_label
                    else:
                        mask_front = mask_front * mk        # get masks of all object in arr front_label (2 object)
                
                t_label = label * mask_front                # noise label with some not correct segmentation
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break
                # noise image have more than 100 pixel
        
        # obj = np.array([obj_setting['exported_objects'][i]['segmentation_class_id'] for i in range(obj_setting['exported_objects'])]).flatten().astype(np.int32)                
        
        # Get the region which has depth!=0 and number of pixels (has object)  > minimum_num_pt
        if (len(obj) == 0):
            return  False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        else:
            tmp = 0
            while 1:
                idx = np.random.randint(0, len(obj))                            # get a random object in the image
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))     # get mask of region that depth not equal 0
                mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))  # sign mask for the pixel which is labeled with its class id / segmentation_class_id
                mask = mask_label * mask_depth
                tmp += 1                                  # only get depth info at pixel labeled of the object random gotten
                if len(mask.nonzero()[0]) >= self.minimum_num_pt:
                    break
                elif tmp>10:
                    return  False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

            #print('{0}/{1}.jpg'.format(self.root, self.list[index])) 
            #print(obj[idx], obj)

            if self.list[index][:5] == 'mixed':
                name_object_masked = self.get_class_name(obj[idx],obj_setting, 'mixed')
                #print(name_object_masked)
                id_json = self.get_id_json_image(name_object_masked, meta)
                vi = []
                for i in id_json:
                    vi.append(meta['objects'][i]['visibility'])
                #print(id_json, vi)
                
                if len(id_json) == 0 or len(vi) == 0 :
                    return False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
                else:
                    vi, id_json = (list(t) for t in zip(*sorted(zip(vi, id_json), reverse=True)))

                #print(id_json, vi)
                for i in id_json:
                    rmin, rmax, cmin, cmax = get_bbox(meta, i)           # bbox for the choosen object
                    if (rmin == rmax) or (cmin == cmax):
                        if i != id_json[-1]:
                            continue
                        else:
                            return False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
                    else: 
                        id_json = i
                        break

            elif self.list[index][:6] == 'single':
                name_object_masked = self.get_class_name(obj[idx],obj_setting, 'single')
                id_json = self.get_id_json_image(name_object_masked, meta)
                if len(id_json) != 1:
                    return False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
                else:
                    id_json = self.get_id_json_image(name_object_masked, meta)[0]
                rmin, rmax, cmin, cmax = get_bbox(meta, id_json)           # bbox for the object in single
                if (rmin == rmax) or (cmin == cmax):

                    return False, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

            img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax] # crop image out of bounding box

            if self.list[index][:5] == 'mixed':
                seed = random.choice(self.single)
                back = np.array(self.trancolor(self.resize(Image.open('{0}/{1}.jpg'.format(self.root, seed))).convert("RGB")))
                back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
                img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
            else:
                img_masked = img

            if self.add_noise and add_front:
                img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

            if self.list[index][:5] == 'mixed':
                img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

            # p_img = np.transpose(img_masked, (1, 2, 0))
            # scipy.misc.imsave('temp/{0}_input.png'.format(index), p_img)
            # scipy.misc.imsave('temp/{0}_label.png'.format(index), mask[rmin:rmax, cmin:cmax].astype(np.int32))
            
            idx_in_class_txt = self.list_of_class.index(name_object_masked[:-4])
            target_r = np.array(meta['objects'][id_json]['quaternion_xyzw'])
            target_r = R.from_quat(target_r).as_matrix()

            # Translation
            target_t = np.array([meta['objects'][id_json]['location']])/100
            
            #noise translate
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                if choose.size == 0:
                    choose = np.pad(choose, (0, self.num_pt - len(choose)), 'constant')
                else: 
                    choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
            """
            with open('name.txt','w') as f:
                        pa = '{0}/{1}.jpg'.format(self.root, self.list[index])
                        n = str(name_object_masked)
                        t = str(id_json)
                        c = str(choose)
                        m = str(mask)
                        print(choose.size)
                        d = depth[rmin:rmax, cmin:cmax].flatten()
                        print(rmin, rmax, cmin, cmax)
                        s = str([rmin, rmax, cmin, cmax])
                        if d.size == 0:
                            print(d)
                            print(d.shape)
                        f.write(pa)    
                        f.write(n+'\n')
                        f.write(t+'\n')
                        f.write(c+'\n')
                        f.write(m+'\n')
                        f.write(s+'\n')
                        d = str(depth[rmin:rmax, cmin:cmax].flatten())
                        
                        f.write(d+'\n')
                        f.close
            """
            
            # Rotation
            # choose point for pointcloud
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])
            
            # Calculate Real Coordinate
            cam_scale = 10000
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            # cloud
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        
            if self.add_noise:
                cloud = np.add(cloud, add_t)

            dellist = [j for j in range(0, len(self.cld['{0}'.format(name_object_masked[:-4])]))]
            if self.refine:
                dellist = random.sample(dellist, len(self.cld['{0}'.format(name_object_masked[:-4])]) - self.num_pt_mesh_large)
            else:
                dellist = random.sample(dellist, len(self.cld['{0}'.format(name_object_masked[:-4])]) - self.num_pt_mesh_small)
            model_points = np.delete(self.cld['{0}'.format(name_object_masked[:-4])], dellist, axis=0)


            target = np.dot(model_points, target_r)
            if self.add_noise:
                target = np.add(target, target_t + add_t)
            else:
                target = np.add(target, target_t)
            
            """
            return   1. 3D cordinate of points choosen
                        2. Index of points choosen
                        3. img of object which masked and normalized
                        4. target (r and t per point)
                        5. model_point (points get from 3D model cad - get equal to number of points at 1.)
                        6. idx indicating type of object in classes.txt
            """
            return  True, \
                    torch.from_numpy(cloud.astype(np.float32)), \
                    torch.LongTensor(choose.astype(np.int32)), \
                    self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                    torch.from_numpy(target.astype(np.float32)), \
                    torch.from_numpy(model_points.astype(np.float32)), \
                    torch.LongTensor([int(idx_in_class_txt)]) #index indicate type of object
                

    def __len__(self):
        return self.length
    
    def get_class_name(self, seg_id, obj, str):
        if str == "mixed":
            class_idx = int((seg_id/12) - 1)
            class_name = obj["exported_objects"][class_idx]["class"]
        elif str == "single":
            class_name = obj["exported_objects"][0]["class"]
        return class_name
    def get_id_json_image(self, name, meta):
        idx = []
        for i in range(len(meta['objects'])):
            if name == meta['objects'][i]['class']:
                idx.append(i)
            else: continue
        return idx
    
    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 960
img_length = 540

def get_bbox(json, idx):
    
    bounding_box = json['objects'][idx]['bounding_box']
    rmin = np.clip(np.floor(bounding_box['top_left'][0]).astype(np.int32), 0, img_length)
    rmax = np.clip(np.ceil(bounding_box['bottom_right'][0]).astype(np.int32),0, img_length)
    cmin = np.clip(np.floor(bounding_box['top_left'][1]).astype(np.int32), 0, img_width)
    cmax = np.clip(np.ceil(bounding_box['bottom_right'][1]).astype(np.int32),0 , img_width)
    return rmin, rmax, cmin, cmax
