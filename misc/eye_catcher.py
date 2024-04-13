from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
from geomloss import SamplesLoss
import ot
import scipy as sp
image_list = []

start_idx = 315 #215
Loss = SamplesLoss("energy")

for i in range(10):
    image = Image.open("/tmp/pycharm_project_165/iphone-space-out/rgb/2x/2_00" + str(start_idx+i) + ".png")
   # image = image.resize((200,200))
    image = torch.tensor(np.array(image)).view([-1, 4]) / 255.
    image = image[:, :3] * image[:, -1:] + (1 - image[:, -1:])
    image_list.append(image)

wsdist_list = []
l2dist_list = []
l2_arr = np.zeros((10,10))
ws_arr = np.zeros((10,10))
sh_arr = np.zeros((10,10))
w_arr  =  np.zeros((10,10))
def compute_wasserstein(xs, ys):
    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(ys, ys)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(40000)
    q = ot.unif(40000)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=True, log=True)
    
    return gw0, log0

for i in range(len(image_list)):
    for j in range(len(image_list)):
        rgb_1 = image_list[i].reshape(-1, 3).cuda()
        rgb_2 = image_list[j].reshape(-1, 3).cuda()

     #   indices = np.random.choice(rgb_1.shape[0], 1000, replace=False)

      #  rgb_1 = rgb_1[indices]
       # rgb_2 = rgb_2[indices]

        shdist = Loss(rgb_1, rgb_2)


        wvec = torch.rand(256, 1, rgb_1.shape[-1]).cuda()
        wvec = wvec / torch.linalg.vector_norm(wvec, dim=2, keepdim=True)

        p = (rgb_1[None, :, :] * wvec).sum(2)
        q = (rgb_2[None, :, :] * wvec).sum(2)

        p, _ = torch.sort(p, dim=1)
        q, _ = torch.sort(q, dim=1)

        wsdist = torch.mean(torch.abs(p - q)**2)
        l2dist = torch.mean(torch.abs(rgb_1 - rgb_2)**2)
        print(f"slice_dist {wsdist}")
     #   gw0, log0 = compute_wasserstein(rgb_1.detach().cpu().numpy(), rgb_2.detach().cpu().numpy())
       # print(f"slice_dist {log0['gw_dist']}")

        l2_arr[i,j] = l2dist
        ws_arr[i, j] = wsdist
        sh_arr[i, j] = shdist
      #  w_arr[i,j] = log0['gw_dist']
# l2_arr = np.abs((l2_arr - np.mean(l2_arr))/np.std(l2_arr))
# ws_arr = np.abs((ws_arr - np.mean(ws_arr))/np.std(ws_arr))

#l2_arr = np.abs((l2_arr)/np.std(l2_arr))
#ws_arr = np.abs((ws_arr)/np.std(ws_arr))

#vmax = l2_arr.max()

ax = sns.heatmap(l2_arr, linewidth=0.5)
plt.xlabel('Image index')
plt.ylabel('Image index')
plt.savefig(f"eye_catcherso/l2_mat.png")
plt.close()

ax2 = sns.heatmap(ws_arr, linewidth=0.5)
plt.xlabel('Image index')
plt.ylabel('Image index')
plt.savefig(f"eye_catcherso/ws_mat.png")
plt.close()

ax2 = sns.heatmap(sh_arr, linewidth=0.5)
plt.xlabel('Image index')
plt.ylabel('Image index')
plt.savefig(f"eye_catcherso/energy_mat.png")
plt.close()

# ax2 = sns.heatmap(w_arr, linewidth=0.5)
# plt.xlabel('Image index')
# plt.ylabel('Image index')
# plt.savefig(f"eye_catcher/w_mat.png")
# plt.close()

# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4,1,0.2]))
#
# sns.heatmap(df, annot=True, cbar=False, ax=axs[0], vmin=vmin)
# sns.heatmap(df2, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax)
#
# fig.colorbar(axs[1].collections[0], cax=axs[2])

for i in range(len(image_list)):
    rgb_1 = image_list[0].reshape(-1,3).cuda()
    rgb_2 = image_list[i].reshape(-1,3).cuda()

    wvec = torch.rand(256, 1, rgb_1.shape[-1]).cuda()
    wvec = wvec / torch.linalg.vector_norm(wvec, dim=2, keepdim=True)

    p = (rgb_1[None, :, :] * wvec).sum(2)
    q = (rgb_2[None, :, :] * wvec).sum(2)

    p, _ = torch.sort(p, dim=1)
    q, _ = torch.sort(q, dim=1)

    wsdist = torch.mean(torch.abs(p - q)**2)
    l2dist = torch.mean(torch.abs(rgb_1 - rgb_2)**2)

    l2dist_list.append(l2dist.detach().cpu().numpy())
    wsdist_list.append(wsdist.detach().cpu().numpy())
    print("test")

plt.plot(l2dist_list, label="Pixel distance")
plt.plot(wsdist_list, label="OT distance")

plt.xlabel('Image index')
plt.ylabel('Distance')
# plt.xlim(0, )
# plt.ylim(0, None)
plt.grid(True)
plt.legend()
plt.savefig(f"eye_catcher/l2_vs_ws.png")





