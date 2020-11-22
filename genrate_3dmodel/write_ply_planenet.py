#from planenet code is adapted for planercnn code

import cv2 
import numpy as np 

WIDTH = 256
HEIGHT = 192

ALL_TITLES = ['PlaneNet']
ALL_METHODS = [('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 0, 2)]

def predict3D(folder, index, image, depth, segmentation, planes, info):


    writePLYFile(folder, index, image, depth, segmentation, planes, info)
    #writePLYFile(options.test_dir, image_index + options.startIndex, segmentationImageBlended, pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], pred_dict['info'][image_index])
    print("done")

def getCameraFromInfo(info):
    camera = {}
    camera['fx'] = info[0]
    camera['fy'] = info[5]
    camera['cx'] = info[2]
    camera['cy'] = info[6]
    camera['width'] = info[16]
    camera['height'] = info[17]
    camera['depth_shift'] = info[18]    
    return camera

def writePLYFile(folder, index, image, depth, segmentation, planes, info):
    imageFilename = str(index) + '_model_texture.png'
    cv2.imwrite(folder + '/' + imageFilename, image)

    width = image.shape[1]
    height = image.shape[0]
    
    numPlanes = planes.shape[0]
    
    camera = getCameraFromInfo(info)

    #camera = getNYURGBDCamera()
    #camera = getSUNCGCamera()

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    XYZ = np.stack([X, Y, Z], axis=2)

    
    #focalLength = 517.97
        
    faces = []
    #minDepthDiff = 0.15
    #maxDepthDiff = 0.3
    #occlusionBoundary = boundaries[:, :, 1]
    betweenRegionThreshold = 0.1
    nonPlanarRegionThreshold = 0.02
    
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = -planes / np.maximum(planesD, 1e-4)    

    croppingRatio = -0.05
    dotThreshold = np.cos(np.deg2rad(30))
    
    for y in range(height - 1):
        for x in range(width - 1):
            if y < height * croppingRatio or y > height * (1 - croppingRatio) or x < width * croppingRatio or x > width * (1 - croppingRatio):
                continue
            
            segmentIndex = segmentation[y][x]
            if segmentIndex == -1:
                continue    

            point = XYZ[y][x]
            #neighborPixels = []
            validNeighborPixels = []
            for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                neighborSegmentIndex = segmentation[neighborPixel[1]][neighborPixel[0]]
                if neighborSegmentIndex == segmentIndex:
                    if segmentIndex < numPlanes:
                        validNeighborPixels.append(neighborPixel)
                    else:
                        neighborPoint = XYZ[neighborPixel[1]][neighborPixel[0]]
                        if np.linalg.norm(neighborPoint - point) < nonPlanarRegionThreshold:
                            validNeighborPixels.append(neighborPixel)
                            pass
                        pass
                else:
                    neighborPoint = XYZ[neighborPixel[1]][neighborPixel[0]]
                    if segmentIndex < numPlanes and neighborSegmentIndex < numPlanes:
                        if (abs(np.dot(planeNormals[segmentIndex], neighborPoint) + planesD[segmentIndex]) < betweenRegionThreshold or abs(np.dot(planeNormals[neighborSegmentIndex], point) + planesD[neighborSegmentIndex]) < betweenRegionThreshold) and np.abs(np.dot(planeNormals[segmentIndex], planeNormals[neighborSegmentIndex])) < dotThreshold:
                            validNeighborPixels.append(neighborPixel)
                            pass
                    else:
                        if np.linalg.norm(neighborPoint - point) < betweenRegionThreshold:
                            validNeighborPixels.append(neighborPixel)
                            pass
                        pass                            
                    pass
                continue
            if len(validNeighborPixels) == 3:
                faces.append((x, y, x + 1, y + 1, x + 1, y))
                faces.append((x, y, x, y + 1, x + 1, y + 1))
            elif len(validNeighborPixels) == 2 and segmentIndex < numPlanes:
                faces.append((x, y, validNeighborPixels[0][0], validNeighborPixels[0][1], validNeighborPixels[1][0], validNeighborPixels[1][1]))
                pass
            continue
        continue
    
    with open(folder + '/' + str(index) + '_model.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile """
        header += imageFilename
        header += """
element vertex """
        header += str(width * height)
        header += """
property float x
property float y
property float z
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in range(height):
            for x in range(width):
                segmentIndex = segmentation[y][x]
                if segmentIndex == -1:
                    f.write("0.0 0.0 0.0\n")
                    continue
                point = XYZ[y][x]
                X = point[0]
                Y = point[1]
                Z = point[2]
                #Y = depth[y][x]
                #X = Y / focalLength * (x - width / 2) / width * 640
                #Z = -Y / focalLength * (y - height / 2) / height * 480
                f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + '\n')
                continue
            continue


        for face in faces:
            f.write('3 ')
            for c in range(3):
                f.write(str(face[c * 2 + 1] * width + face[c * 2]) + ' ')
                continue
            f.write('6 ')                     
            for c in range(3):
                f.write(str(float(face[c * 2]) / width) + ' ' + str(1 - float(face[c * 2 + 1]) / height) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return  


def evaluatePlanes(options):
 
    

     
    
    for image_index in range(options.visualizeImages):

        if options.applicationType == 'grids':
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', pred_dict['image'][image_index])
            segmentation = predictions[0]['segmentation'][image_index]
            #segmentation = np.argmax(np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2), -1)
            segmentationImage = drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes)
            #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(0) + '.png', segmentationImage)
            segmentationImageBlended = (segmentationImage * 0.7 + pred_dict['image'][image_index] * 0.3).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(0) + '.png', segmentationImageBlended)
            continue

            
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', pred_dict['image'][image_index])
        
        info = pred_dict['info'][image_index]

        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if 'pixelwise' in options.methods[method_index][1]:
                continue
            allSegmentations = pred_dict['segmentation'][image_index]
            segmentation = np.argmax(allSegmentations, axis=-1)
            #segmentation = np.argmax(np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2), -1)
            segmentationImage = drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', segmentationImage)
            segmentationImageBlended = (segmentationImage * 0.7 + pred_dict['image'][image_index] * 0.3).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(method_index) + '.png', segmentationImageBlended)

            segmentationImageBlended = np.minimum(segmentationImage * 0.3 + pred_dict['image'][image_index] * 0.7, 255).astype(np.uint8)

            if options.imageIndex >= 0:
                for planeIndex in range(options.numOutputPlanes):
                    cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                    continue
                
                if options.applicationType == 'logo_video':
                    copyLogoVideo(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index], textureType='logo')
                elif options.applicationType == 'wall_video':
                    if options.wallIndices == '':
                        print('please specify wall indices')
                        exit(1)
                        pass
                    wallIndices = [int(value) for value in options.wallIndices.split(',')]
                    copyLogoVideo(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index], textureType='wall', wallInds=wallIndices)
                elif options.applicationType == 'ruler':
                    if options.startPixel == '' or options.endPixel == '':
                        print('please specify start pixel and end pixel')
                        exit(1)
                        pass                    
                    startPixel = tuple([int(value) for value in options.startPixel.split(',')])
                    endPixel = tuple([int(value) for value in options.endPixel.split(',')])
                    addRulerComplete(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index], startPixel=startPixel, endPixel=endPixel, fixedEndPoint=True, numFrames=1000)
                elif options.applicationType == 'logo_texture':
                    resultImage = copyLogo(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index])
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_result.png', resultImage)
                elif options.applicationType == 'wall_texture':
                    if options.wallIndices == '':
                        print('please specify wall indices')
                        exit(1)
                        pass                    
                    wallIndices = [int(value) for value in options.wallIndices.split(',')]
                    resultImage = copyWallTexture(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index], wallPlanes=wallIndices)
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_result.png', resultImage)                    
                elif options.applicationType == 'TV':
                    if options.wallIndices == '':
                        print('please specify wall indices')
                        exit(1)
                        pass                    
                    wallIndices = [int(value) for value in options.wallIndices.split(',')]
                    copyLogoVideo(options.textureImageFilename, options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index], textureType='TV', wallInds=wallIndices)
                elif options.applicationType == 'pool':
                    print('dump')
                    newPlanes = []
                    newSegmentation = np.full(segmentation.shape, -1)
                    newPlaneIndex = 0
                    planes = pred_dict['plane'][image_index]
                    for planeIndex in range(options.numOutputPlanes):
                        mask = segmentation == planeIndex
                        if mask.sum() > 0:
                            newPlanes.append(planes[planeIndex])
                            newSegmentation[mask] = newPlaneIndex
                            newPlaneIndex += 1
                            pass
                        continue

                    np.save('pool/dump/' + str(image_index + options.startIndex) + '_planes.npy', np.stack(newPlanes, axis=0))
                    #print(global_gt['non_plane_mask'].shape)
                    np.save('pool/dump/' + str(image_index + options.startIndex) + '_segmentation.npy', newSegmentation)
                    cv2.imwrite('pool/dump/' + str(image_index + options.startIndex) + '_image.png', pred_dict['image'][image_index])
                    depth = pred_dict['depth'][image_index]
                    np.save('pool/dump/' + str(image_index + options.startIndex) + '_depth.npy', depth)
                    info = pred_dict['info'][image_index]
                    #normal = calcNormal(depth, info)
                    #np.save('test/' + str(image_index + options.startIndex) + '_normal.npy', normal)
                    np.save('pool/dump/' + str(image_index + options.startIndex) + '_info.npy', info)
                    exit(1)
                else:
                    print('please specify application type')
                    np_mask = (segmentation == options.numOutputPlanes).astype(np.float32)
                    np_depth = pred_dict['np_depth'][image_index].squeeze()
                    np_depth = cv2.resize(np_depth, (np_mask.shape[1], np_mask.shape[0]))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_np_depth_pred_' + str(method_index) + '.png', drawDepthImage(np_depth * np_mask))
                    # folder, \  - directory           - done
                    # index, \  - idx number of image     - done
                    # image, \  - segmentationImageBlended
                    # depth, \    - pred_dict['depth'][image_index]  - done
                    # segmentation, \  - segmentation
                    # planes, \ - pred_dict['plane'][image_index]
                    # info  - pred_dict['info'][image_index]       - done

                    writePLYFile(options.test_dir, image_index + options.startIndex, segmentationImageBlended, pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], pred_dict['info'][image_index])
                    pass
                exit(1)
                pass
            continue
        continue

    writeHTML(options)
    return


def getResults(options):
    checkpoint_prefix = 'checkpoint/'

    methods = options.methods
    predictions = []

    if os.path.exists(options.result_filename) and options.useCache == 1:
        predictions = np.load(options.result_filename)
        return predictions
    

    for method_index, method in enumerate(methods):
        if len(method) < 4 or method[3] < 2:
            continue
        if method[0] == '':
            continue
        
        if 'ds0' not in method[0]:
            options.deepSupervisionLayers = ['res4b22_relu', ]
        else:
            options.deepSupervisionLayers = []
            pass
        options.predictConfidence = 0
        options.predictLocal = 0
        options.predictPixelwise = 1
        options.predictBoundary = int('pb' in method[0])
        options.anchorPlanes = 0
        if 'ps' in method[0]:
            options.predictSemantics = 1
        else:
            options.predictSemantics = 0
            pass
        if 'crfrnn' in method[0]:
            options.crfrnn = 10
        else:
            options.crfrnn = 0
            pass
            
        if 'ap1' in method[0]:
            options.anchorPlanes = 1            
            pass
        
        options.checkpoint_dir = checkpoint_prefix + method[0]
        print(options.checkpoint_dir)
        
        options.suffix = method[1]

        method_names = [previous_method[0] for previous_method in methods[:method_index]]

        if options.customImageFolder != '':
            print('make predictions on custom images')
            pred_dict = getPredictionCustom(options)
        elif options.dataFolder != '':
            print('make predictions on ScanNet images')            
            pred_dict = getPredictionScanNet(options)
        else:
            print('please specify customImageFolder or dataFolder')
            exit(1)
            pass
        
        predictions.append(pred_dict)
        continue
    #np.save(options.test_dir + '/curves.npy', curves)
    results = predictions

    #print(results)
    
    if options.useCache != -1:
        np.save(options.result_filename, results)
        pass
    pass
    
    return results




def getPredictionCustom(options):
    tf.reset_default_graph()
    
    options.batchSize = 1

    img_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
    training_flag = tf.constant(False, tf.bool)

    options.gpu_id = 0
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)

    var_to_restore = tf.global_variables()


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    width_high_res = 640
    height_high_res = 480
                

    #image_list = glob.glob('../my_images/*.jpg') + glob.glob('../my_images/*.png') + glob.glob('../my_images/*.JPG')
    #image_list = glob.glob('../my_images/TV/*.jpg') + glob.glob('../my_images/TV/*.png') + glob.glob('../my_images/TV/*.JPG')
    #image_list = glob.glob('../my_images/TV/*.jpg') + glob.glob('../my_images/TV/*.png') + glob.glob('../my_images/TV/*.JPG')
    image_list = glob.glob(options.customImageFolder + '/*.jpg') + glob.glob(options.customImageFolder + '/*.png') + glob.glob(options.customImageFolder + '/*.JPG')
    options.visualizeImages = min(options.visualizeImages, len(image_list))
    
    pred_dict = {}
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        try:
            predDepths = []
            predPlanes = []
            predSegmentations = []
            predSemantics = []            
            predNonPlaneDepths = []
            predNonPlaneNormals = []            
            predNonPlaneMasks = []
            predBoundaries = []
            images = []
            infos = []
            for index in range(min(options.startIndex + options.numImages, len(image_list))):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()
                
                print(('image', index))
                
                img_ori = cv2.imread(image_list[index])
                images.append(img_ori)
                img = cv2.resize(img_ori, (WIDTH, HEIGHT))
                img = img.astype(np.float32) / 255 - 0.5
                img = np.expand_dims(img, 0)
                global_pred = sess.run(global_pred_dict, feed_dict={img_inp: img})

                if index < options.startIndex:
                    continue                


                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]
                
                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]
                
                #if global_gt['info'][0][19] > 1 and global_gt['info'][0][19] < 4 and False:
                #pred_np_n = calcNormal(pred_np_d.squeeze(), global_gt['info'][0])
                #pass


                #pred_b = global_pred['boundary'][0]
                predNonPlaneMasks.append(pred_np_m)                    
                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)
                    
                all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)

                info = np.zeros(20)
                if options.estimateFocalLength:
                    focalLength = estimateFocalLength(img_ori)
                    info[0] = focalLength
                    info[5] = focalLength
                    info[2] = img_ori.shape[1] / 2
                    info[6] = img_ori.shape[0] / 2
                    info[16] = img_ori.shape[1]
                    info[17] = img_ori.shape[0]
                    info[10] = 1
                    info[15] = 1
                    info[18] = 1000
                    info[19] = 5
                else:
                    info[0] = 2800.71
                    info[2] = 1634.45
                    info[5] = 2814.01
                    info[6] = 1224.18
                    info[16] = img_ori.shape[1]
                    info[17] = img_ori.shape[0]
                    info[10] = 1
                    info[15] = 1
                    info[18] = 1000
                    info[19] = 5
                    pass

                # print(focalLength)
                # cv2.imwrite('test/image.png', ((img[0] + 0.5) * 255).astype(np.uint8))
                # cv2.imwrite('test/segmentation.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                # exit(1)
                infos.append(info)
                width_high_res = img_ori.shape[1]
                height_high_res = img_ori.shape[0]
                
                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

                pred_np_d = np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)
                all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

                all_segmentations = np.stack([cv2.resize(all_segmentations[:, :, planeIndex], (width_high_res, height_high_res)) for planeIndex in range(all_segmentations.shape[-1])], axis=2)
                
                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(height_high_res * width_high_res), segmentation.reshape(-1)].reshape(height_high_res, width_high_res)

                if 'semantics' in global_pred:
                    #cv2.imwrite('test/semantics.png', drawSegmentationImage(np.argmax(global_pred['semantics'][0], axis=-1)))
                    #exit(1)
                    predSemantics.append(np.argmax(global_pred['semantics'][0], axis=-1))
                else:
                    predSemantics.append(np.zeros((HEIGHT, WIDTH)))
                    pass
                                         
                predDepths.append(pred_d)
                predPlanes.append(pred_p)
                predSegmentations.append(all_segmentations)
                continue
            pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['depth'] = np.array(predDepths)
            #pred_dict['semantics'] = np.array(predSemantics)                        
            pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            #pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            pred_dict['image'] = np.array(images)
            pred_dict['info'] = np.array(infos)
            #pred_dict['boundary'] = np.array(predBoundaries)
            pass
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return pred_dict



if __name__=='__main__':
    info = np.array([1.82e+03, 0.00e+00, 1.63e+03, 0.00e+00,\
     0.00e+00, 1.82e+03, 1.22e+03, 0.00e+00, 0.00e+00, 0.00e+00, \
     1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 3.26e+03, 2.45e+03,\
      1.00e+03,5.00e+00])
    image = cv2.imread("single_rgb_sample/12/12_segmentation_0_final.png") #x,x,3
    depth = cv2.imread("single_rgb_sample/12/12_depth_0_final_ori.png",0) #x,x
    segmentation = cv2.imread("single_rgb_sample/12/12_segmentation_0_final.png",0) #change it
    planes =  np.load("single_rgb_sample/12/12_plane_masks_0.npy") #change if its not working
    folder = "predict3fol"
    index = 12
    predict3D(folder, index, image, depth, segmentation, planes, info)
        
#todo
# try to add focal length
# try to do with rgb based one 