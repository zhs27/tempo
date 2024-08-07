import numpy as np
import torch
from pytorch_wavelets import DWTForward, DWTInverse

import numpy as np
import torch
from torchvision.utils import save_image


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
DWT_DEFAULT_PARAMS = {'mode': 'zero', 'wave': 'haar', 'J': 1}
softmax = torch.nn.Softmax(dim=-1)

class CartoonX:
    def __init__(
            self, 
            model,
            noise_bs,
            optim_steps,
            lr,
            l1lambda,
            mask_init,
            obfuscation='uniform',
            maximize_label=False,
            dwt_params=DWT_DEFAULT_PARAMS,
            device=DEVICE):
        """
        args:
           model: nn.Module classifier to be explained
           noise_bs: int number of noise perturbation samples
           optim_steps: int number of optimization steps
           lr: float learning rate for mask
           l1lambda: float l1 wavelet coefficient multiplier
           obfuscation: str "gaussian" or "uniform"
           maximize_label: bool - whether to maximize the label probability
           mask_init: List mask on wavelet coefficients (comes as list with submasks)
           dwt_params: dict paramters for DWT
           device: str cpu or gpu
        """
        self.model = model
        self.noise_bs = noise_bs
        self.optim_steps = optim_steps
        self.lr = lr
        self.l1lambda = l1lambda
        self.obfuscation = obfuscation
        self.maximize_label = maximize_label
        self.mask_init = mask_init
        self.dwt_params=dwt_params
        self.device = device

        self.forward_dwt = DWTForward(**dwt_params).to(device)
        self.inverse_dwt = DWTInverse(mode=dwt_params['mode'], wave=dwt_params['wave']).to(device)
        self.get_perturbation = None # this method will be assigned in compute_obfuscation_strategy

    def __call__(self, x, target, xtocartoonx = None):
        """
        args:
            x: torch.Tensor of shape (bs,c,h,w)
            target: torch.Tensor of shape (bs,)
        """
        '''
        assert len(x.shape)==4
        assert x.requires_grad == False
        '''
        # Initialize optimization loss tracking
        l1wavelet_loss = []
        distortion_loss = []

        if xtocartoonx != None:
            ret = []
            for i in x:
                if i in xtocartoonx:
                    ret.append(xtocartoonx[i])
                else:
                    break
            if len(ret) == len(x):
                print(1)
                return ret
        
        #apply dwt on all images
        yl = []
        yh = []
        m_yl = []
        m_yh = []
        optpara = []
        for i in x:
            y1,y2 = self.forward_dwt(i)
            self.compute_obfuscation_strategy(y1, y2)
            m_y1,m_y2 = self.get_init_mask(y1,y2)
            optpara += [m_y1]+m_y2
            yl.append(y1)
            yh.append(y2)
            m_yl.append(m_y1)
            m_yh.append(m_y2)
        
        # compute obfuscation strategy

        
        '''
        # Get wavelet coefficients of colored image 
        # (yl are low pass coefficients, yh are high pass coeffcients)
        # yl is a tensor and yh is a list of tensors (see pytorch wavelets doc)
        yl, yh = self.forward_dwt(x)

        # Get wavelet coefficients of grayscale image
        yl_gray, yh_gray = self.forward_dwt(x.sum(dim=1, keepdim=True)/3)
        
        # Compute obfuscation strategy
        self.compute_obfuscation_strategy(yl, yh)

        # Initialize mask on wavelet coefficients yl and yh
        m_yl, m_yh = self.get_init_mask(yl, yh)
        
        '''
        # Get total number of mask entries
        with torch.no_grad():
            num_mask = 0
            for i,j in zip(m_yl, m_yh):
                num_mask += i.view(i.size(0), -1).size(-1) + sum([m.view(m.size(0), -1).size(-1) for m in j])
        
        
        # Initialize optimizer

        opt = torch.optim.Adam(optpara, lr=self.lr)
        
        # Get reference output for distortion
        if self.maximize_label:
            out_x =  torch.ones((x.size(0),),
                                requires_grad=False,
                                dtype=torch.float32,
                                device=self.device)
        else: 
            out_x = self.get_model_output(x, target)
       
        # Optimize wavelet mask with projected GD
        for i in range(self.optim_steps):
            obf_x = []
            for myh,myl,yll,yhh in zip(m_yh,m_yl,yl,yh):
                # Get perturbation on wavelet coefficients yl and yh
                p_yl, p_yh = self.get_perturbation()
                # Obfuscate wavelet coefficients yl
                obf_yl = myl.unsqueeze(1) * yll.unsqueeze(1) + (1 - myl.unsqueeze(1)) * p_yl
                # Obfuscate wavelet coefficients yh
                obf_yh = []
                for y, m, p in zip(yhh, myh, p_yh): obf_yh.append((m.unsqueeze(1)*y.unsqueeze(1)+(1-m.unsqueeze(1))*p))
                # Get obfuscation in pixel space by applying inverse dwt and projecting into [0,1]
                obf_x.append(self.inverse_dwt((obf_yl.reshape(-1, *obf_yl.shape[2:]), [o.reshape(-1,*o.shape[2:]) for o in obf_yh])).clamp(0,1))
            # Get model output for obfuscation (need to have one copy for each noise perturbation sample)
            obf_x = torch.stack(obf_x)
            targets_copied = target
            out_obf = self.get_model_output(obf_x, targets_copied)                        
            # Compute model output distortion between x and obf_x
            distortion_batch = torch.mean((out_x.unsqueeze(1) - out_obf)**2, dim=-1)
            distortion = distortion_batch.sum()
            # Compute l1 norm of wavelet coefficients

            l1waveletcoefs = 0
            for m in m_yl: l1waveletcoefs += m.abs().sum() 
            for m in m_yh:
                for n in m: 
                    l1waveletcoefs += n.abs().sum()

            l1waveletcoefs /= num_mask
            
            # Log losses
            distortion_loss.append(distortion_batch.detach().clone().cpu().numpy())
            l1wavelet_loss.append(l1waveletcoefs.item())
                
            # Compute optiimization loss
            loss = distortion + self.l1lambda * l1waveletcoefs 
            # Perform optimization step
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            # Project masks into [0,1]
            with torch.no_grad():
                for m in m_yl:
                    m.clamp_(0,1)
                for m in m_yh: 
                    for n in m: n.clamp_(0,1)


        # Invert wavelet coefficient mask back to pixel space as grayscale images
        #cartoonx = self.inverse_dwt((m_yl.detach()*yl_gray, [m.detach()*y for m,y in zip(m_yh, yh_gray)])).clamp(0,1)
        cartoonx = []
        for myl,myh,l,h in zip(m_yl,m_yh,yl,yh):
            cartoonx_per_rgb = self.inverse_dwt(
                    (myl.detach()*l[:,0,:,:].unsqueeze(1), 
                     [m.detach()*y[:,0,:,:,:].unsqueeze(1) for m,y in zip(myh, h)]
                    )
                )
            
            cartoonx.append(cartoonx_per_rgb.clamp(0,1))
        '''
        for i,j in zip(x, cartoonx):
            xtocartoonx.update(i = j)
        '''
        
        '''
        cartoonx_per_rgb = [
                self.inverse_dwt(
                    (m_yl.detach()*yl[:,i,:,:].unsqueeze(1), 
                     [m.detach()*y[:,i,:,:,:].unsqueeze(1) for m,y in zip(m_yh, yh)]
                    )
                ) for i in [0]
        ]
        '''
        #assert tuple(cartoonx.shape)==tuple(x.shape), cartoonx.shape

        '''
        # Get a dictionary with losses, mask statistics, and final mask
        history = {'mask': (m_yl.detach(), [m.detach() for m in m_yh]),
                   'distortion': distortion_loss,
                   'l1wavelet': l1wavelet_loss
                  }
        '''
        return cartoonx 
    
    def compute_obfuscation_strategy(self, yl, yh):
        # Get std and mean of yl wavelet coefficients per image
        std_yl = torch.std(yl, dim=[1,2,3]).reshape(yl.size(0),1,1,1,1)
        mean_yl = torch.mean(yl, dim=[1,2,3]).reshape(yl.size(0),1,1,1,1)
        # get std and mean of yh wavelet coefficients per image
        std_yh, mean_yh = [], []
        for y in yh: 
            std_yh.append(torch.std(y, dim=[1,2,3,4]).reshape(y.size(0),1,1,1,1,1))
            mean_yh.append(torch.mean(y, dim=[1,2,3,4]).reshape(y.size(0), 1,1,1,1,1))
        if self.obfuscation == 'gaussian':
            def get_perturbation():
                # Perturbation for yl wavelet coefficients
                pert_yl = std_yl * torch.randn((yl.size(0), self.noise_bs, *yl.shape[1:]),
                                               dtype=torch.float32,
                                               device=self.device,
                                               requires_grad=False) + mean_yl
                # Perturbation for yh wavelet coefficients
                pert_yh = []
                for y, std, mean in zip(yh, std_yh, mean_yh):
                    p = std * torch.randn((y.size(0), self.noise_bs, *y.shape[1:]), 
                                                  dtype=torch.float32, 
                                                  device=self.device,
                                                  requires_grad=False) + mean
                    pert_yh.append(p)
                return (pert_yl, pert_yh)
        elif self.obfuscation == 'uniform':
            def get_perturbation():
                # Perturbation for yl wavelet coefficients
                pert_yl = torch.rand((yl.size(0), self.noise_bs, *yl.shape[1:]),
                                     dtype=torch.float32,
                                     device=self.device,
                                     requires_grad=False) * (2 * std_yl) + (mean_yl - std_yl)
                
                # Perturbation for yh wavelet coefficients
                pert_yh = []
                for y, std, mean in zip(yh, std_yh, mean_yh):
                    p = torch.rand((y.size(0), self.noise_bs, *y.shape[1:]),
                                   dtype=torch.float32,
                                   device=self.device,
                                   requires_grad=False) * (2 * std) + (mean - std)
                    pert_yh.append(p)
                return (pert_yl, pert_yh)
        elif self.obfuscation == 'zeros':
            def get_perturbation():
                pert_yl = torch.zeros((yl.size(0),self.noise_bs,*yl.shape[1:]),
                                      dtype=torch.float32, 
                                      device=self.device,
                                      requires_grad=False)
                pert_yh = []
                for y in yh: pert_yh.append(torch.zeros((y.size(0),self.noise_bs,*y.shape[1:]), 
                                                       dtype=torch.float32,
                                                       device=self.device,
                                                       requires_grad=False))
                return (pert_yl, pert_yh)
        else:
            raise NotImplementedError('Only uniform, gaussian, and zero perturbations were implemented.')

        self.get_perturbation = get_perturbation
        
    def get_init_mask(self, yl, yh):
        if self.mask_init == 'ones':
            # Get all ones mask for yl coefficients
            m_yl = torch.ones((yl.size(0), 1, *yl.shape[2:]), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get all ones mask for yh coefficients
            m_yh = []
            for y in yh:
                m_yh.append(torch.ones((y.size(0), 1, *y.shape[2:]), 
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))
        elif type(self.mask_init) == float or type(self.mask_init) == int:
            # Get constant mask for yl coefficients
            m_yl = torch.full((yl.size(0), 1, *yl.shape[2:]), self.mask_init,
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get constant mask for yh coefficients
            m_yh = []
            for y in yh:
                m_yh.append(torch.full((y.size(0), 1, *y.shape[2:]), self.mask_init,
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))
        elif self.mask_init == 'zeros':
            # Get all zeros mask for yl coefficients
            m_yl = torch.zeros((yl.size(0), 1, *yl.shape[2:]), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get all zeros mask for yh coefficients
            m_yh = []
            for y in yh:
                m_yh.append(torch.zeros((y.size(0), 1, *y.shape[2:]), 
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))
        elif type(self.mask_init) == tuple:
            m_yl, m_yh = self.mask_init
        else:
            raise ValueError('Need to pass string with type of mask or entire initialization mask')
        
        return m_yl, m_yh


    def get_model_output(self, x, target):
        out,_ = self.model(x)
        ret = []   
        for i,j in zip(out,target):
            ret.append(i[j])
        ret = torch.stack(ret)
        
        return ret


