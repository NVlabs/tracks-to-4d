import torch
import imageio
import numpy as np
import cv2 

import matplotlib as mpl
import glob
mpl.use('Agg')
import matplotlib.cm as cm
import colorsys

def add_point_cloud( points,colors, str_to_save):
    np.set_printoptions(threshold=np.inf)
    str_to_save+="buffer_geometry = new THREE.BufferGeometry();\n"
    global_rotation=torch.tensor([[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])
    str_to_save+="buffer_geometry.attributes.position = new THREE.BufferAttribute( new Float32Array(%s), 3 );\n"%(np.array2string((global_rotation@points).T.flatten().numpy(),separator=',',precision=3).replace('\n', ''))
    str_to_save+="buffer_geometry.attributes.color = new THREE.BufferAttribute( new Float32Array(%s), 3 );\n"%(np.array2string(colors.T.flatten().numpy(),separator=',',precision=3).replace('\n', ''))
    str_to_save+="material = new THREE.PointsMaterial( { size: 0.5, vertexColors: true } );\n"
    str_to_save+="point_cloud = new THREE.Points( buffer_geometry, material );\n"
    str_to_save+="point_cloud.visible=false;\n"
    str_to_save+="scene.add(point_cloud);\n"
    str_to_save+="point_clouds  .push(point_cloud);\n"
    return str_to_save



def add_camera( camera_orientation,camera_center, str_to_save,im0):
    np.set_printoptions(threshold=np.inf)
    im_width=im0.shape[1]//2
    im_height=im0.shape[0]//2

    im0_resize=cv2.resize(im0[:,:,:3],(im_width,im_height))


    str_to_save+=" // Create pixel data (50x50, 3 channels for RGB)\nwidth = %d;\nheight = %d;\npixelData = new Uint8Array(width * height * 3); // Array to hold pixel data\n"%(im_width,im_height)
        
        
    str_to_save+="pixelData= new Uint8Array(%s);\n"%np.array2string(im0_resize.flatten(),separator=',').replace('\n', '')
    
    width_world_units=10*im_width/im_height
    height_world_units=10
    focal_world_units=6
    global_rotation=np.array([[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])
    camera_center=global_rotation@camera_center
    camera_orientation=global_rotation@camera_orientation
    c_x=camera_center[0]
    c_y=camera_center[1]
    c_z=camera_center[2]
    
    square1=camera_orientation@np.array([-width_world_units/2,-height_world_units/2, focal_world_units ])+camera_center
    square2=camera_orientation@np.array([width_world_units/2, -height_world_units/2, focal_world_units])+camera_center
    square3=camera_orientation@np.array([width_world_units/2, height_world_units/2,focal_world_units])+camera_center
    square4=camera_orientation@np.array([-width_world_units/2, height_world_units/2,focal_world_units])+camera_center





    str_to_save+=	"texture = new THREE.DataTexture(pixelData, width, height, THREE.RGBFormat);\n"
    str_to_save+=	"texture.needsUpdate = true; // Important: notify Three.js that texture has changed\n"

    str_to_save+=	"// Create a flat square base (plane geometry)\n"
    str_to_save+=	"baseGeometry = new THREE.PlaneGeometry(%3.3f, %3.3f); // Square with size 5x5\n"%(width_world_units,height_world_units)
    str_to_save+=	"baseMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });\n"
    str_to_save+=	"base = new THREE.Mesh(baseGeometry, baseMaterial);\n"

    str_to_save+=	"// Rotate the base to lie flat on the xz-plane\n"
    str_to_save+=	"//base.rotation.x = Math.PI / 2; // Rotate to match the pyramid base\n"
    str_to_save+=	"base.translateZ(%f);\n"%focal_world_units
   
    str_to_save+=   "Rot = new THREE.Matrix4(); \n"
    str_to_save+=   "Rot.set( %f, %f, %f,%f , \n"%(camera_orientation[0,0],camera_orientation[0,1],camera_orientation[0,2],camera_center[0])
    str_to_save+= 	         " %f, %f, %f,%f,  \n"%(camera_orientation[1,0],camera_orientation[1,1],camera_orientation[1,2],camera_center[1])
    str_to_save+= 	            " %f, %f, %f, %f, \n "%(camera_orientation[2,0],camera_orientation[2,1],camera_orientation[2,2],camera_center[2])
    str_to_save+=   "  0.0,0.0,0.0,1.0 );\n"
    str_to_save+= 	"base.applyMatrix4(Rot);\n"
    str_to_save+=	"// Add the base to the scene\n"

    str_to_save+=	"scene.add(base);\n"
    str_to_save+=	"base.visible=false\n"
    str_to_save+=	"// Create the top of the pyramid (a point above the base)\n"
    str_to_save+=	"tops = new THREE.Vector3(%3.3f, %3.3f, %3.3f); // The apex of the pyramid, 5 units above the center\n"%(c_x,c_y,c_z)

    str_to_save+=	"// Create line segments for the edges of the pyramid\n"
    str_to_save+=	"edgePoints = [ new THREE.Vector3(%3.3f, %3.3f,%3.3f )"%(square1[0],square1[1],square1[2])+", new THREE.Vector3(%3.3f, %3.3f,%3.3f )"%(square2[0],square2[1],square2[2])+",new THREE.Vector3(%3.3f, %3.3f,%3.3f)"%(square3[0],square3[1],square3[2])+",new THREE.Vector3(%3.3f, %3.3f,%3.3f),];\n"%(square4[0],square4[1],square4[2])

    str_to_save+=   "// Create geometry for the pyramid edges (lines from top to each corner of the square base)\n"
    str_to_save+=   "lineMaterial = new THREE.LineBasicMaterial({ color:  0x535353,linewidth:0.001,transparent:true,opacity:0.3 }); // Red color for the pyramid edges\n"
    str_to_save+=   "edges = new THREE.Group();\n"

    str_to_save+=   "edgePoints.forEach((point) => {const geometry = new THREE.BufferGeometry().setFromPoints([tops, point]);const line = new THREE.Line(geometry, lineMaterial);edges.add(line);});\n"
    str_to_save+=   "// Create lines for the square base (edges of the square) \n"
    str_to_save+=   "squareEdges = [new THREE.Vector3(%3.3f, %3.3f,%3.3f )"%(square1[0],square1[1],square1[2])+", new THREE.Vector3(%3.3f, %3.3f, %3.3f)"%(square2[0],square2[1],square2[2])+",new THREE.Vector3(%3.3f, %3.3f, %3.3f)"%(square2[0],square2[1],square2[2])+", new THREE.Vector3(%3.3f, %3.3f, %3.3f )"%(square3[0],square3[1],square3[2])+",new THREE.Vector3(%3.3f, %3.3f, %3.3f )"%(square3[0],square3[1],square3[2])+", new THREE.Vector3(%3.3f, %3.3f, %3.3f ),new THREE.Vector3(%3.3f, %3.3f, %3.3f )"%(square4[0],square4[1],square4[2],square4[0],square4[1],square4[2])+", new THREE.Vector3(%3.3f, %3.3f, %3.3f )];\n"%(square1[0],square1[1],square1[2])

    str_to_save+=   "// Create geometry for the square edges\n"
    str_to_save+=   "for (let i = 0; i < squareEdges.length; i += 2) {\n"
    str_to_save+=   "     geometry = new THREE.BufferGeometry().setFromPoints([squareEdges[i], squareEdges[i + 1]]);\n"
    str_to_save+=   "     line = new THREE.Line(geometry, lineMaterial);\n"
    str_to_save+=   "     edges.add(line);\n"
    str_to_save+=   "}\n"
    str_to_save+=   "// Add the edges to the scene\n"
    str_to_save+=   "scene.add(edges);\n"
    str_to_save+=   "bases_all.push (base);\n"
    str_to_save+=   "edges_all.push (edges);\n"
    
    


    return str_to_save

def process_sequence(seq,dataset_path,validation_outputs,logs_folder):
    #
    
    num_cams=validation_outputs[seq]['tracks'].shape[-1]
    data=validation_outputs

    numpoints=data[seq]["tracks"].shape[1]
    
    min_X=data[seq]["tracks"][0,:,0,0].min()

    min_Y=data[seq]["tracks"][0,:,1,0].min()

    x_values=(data[seq]["tracks"][0,:,0,0]-min_X)/((data[seq]["tracks"][0,:,0,0]-min_X).max())*1.5
    x_values[x_values>1]=1
    y_values=(data[seq]["tracks"][0,:,1,0]-min_Y)/((data[seq]["tracks"][0,:,1,0]-min_Y).max())*1.5
    y_values[y_values>1]=1
    rgbs=[]
    for i in range(numpoints):
        rgbs.append(colorsys.hsv_to_rgb(x_values[i].item(),y_values[i].item(),1))

    rgb = cm.rainbow(np.linspace(0, 1, numpoints))
    rgb = torch.from_numpy(np.stack(rgbs))
    ims_points=[]
    for i in range(num_cams):
        K=data[seq]['GT_Ks'][0,i].numpy()
        import matplotlib.pyplot as plt

        plt.figure()
        im=imageio.imread("/".join(dataset_path.split("/")[:-1])+"/view_data/%s/resized_video/%03d.jpg"%(seq,i))

        plt.imshow(im)
        observed=data[seq]["tracks"][0,:numpoints,2,i]>0
        plt.scatter(data[seq]["tracks"][0,:numpoints,0,i][observed]*K[0,0]+K[0,2],data[seq]["tracks"][0,:numpoints,1,i][observed]*K[1,1]+K[1,2],c=rgb[observed],s=20)
        plt.xlim(0,im.shape[1])
        plt.ylim(im.shape[0],0)
        plt.axis('off')
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
        ims_points.append(imageio.imread("temp.png"))
        plt.close("all")



    with open('html_templates/template_close.txt', 'r') as file:
        template_close = file.read()


    with open('html_templates/template_open.txt', 'r') as file:
        template_open = file.read()

    
    str_to_save="bases_all=[];\n numInstances=%d;\nedges_all=[];\n"%num_cams

    for camera_ind in range(0,num_cams,1):
    
        camera_orientation=validation_outputs[seq]['rotation_params'][0,camera_ind].numpy()
        camera_center=validation_outputs[seq]['translation_params'][0,camera_ind].numpy()
    
        im0=ims_points[camera_ind]
        str_to_save=add_camera(camera_orientation,camera_center, str_to_save,im0)


    for camera_ind in range(0,num_cams,1):
        cur_points3d=validation_outputs[seq]['points3D'][0,camera_ind]
        str_to_save=add_point_cloud(cur_points3d,rgb.T,str_to_save)

    writer = imageio.get_writer('%s/output_%s.mp4'%(logs_folder,seq), fps=20)

    for im in ims_points:
        writer.append_data(im)
    writer.close()
    with open('%s/output_%s.html'%(logs_folder,seq), 'w') as file:
        file.write(template_open+str_to_save+template_close)


def make_vizualizations(model_outputs,dataset_path,logs_folder):
    
    folders=glob.glob("%s/*.npz"%dataset_path)
    for folder in folders:
        process_sequence(folder.split("/")[-1][:-4],dataset_path,model_outputs,logs_folder)
       