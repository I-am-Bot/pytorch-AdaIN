import torch
import torch.nn as nn

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s

    """
    1/24/2023 added: non-targeted attack
    """
    def calc_adv_loss(self, input_mean, input_std, target_mean, target_std):
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def adv_forward(self, content, style):
        x_adv = style.detach() + 0.001 * torch.randn(style.shape).cuda().detach()
        # import ipdb
        # ipdb.set_trace()
        epsilon = 16.0 / 255.0
        alpha = 1.6 / 255.0
        # non-target
        style_feats = self.encode_with_intermediate(1 - style)

        for _step in range(50):
            print(_step)
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_feats = self.encode_with_intermediate(x_adv)
                # import ipdb
                # ipdb.set_trace()
                
                # adv_mean, adv_std = calc_mean_std(adv_feats[1])
                # target_mean, target_std = calc_mean_std(style_feats[1])
                loss_adv = 0

                for i in range(0, 4):
                    adv_mean, adv_std = calc_mean_std(adv_feats[i])
                    target_mean, target_std = calc_mean_std(style_feats[i])

                    loss_adv += self.calc_adv_loss(target_mean.detach(), target_std.detach(), adv_mean, adv_std)
                
            grad = torch.autograd.grad(loss_adv, [x_adv])[0]
            x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())
            # x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())

            x_adv = torch.min(torch.max(x_adv, style - epsilon), style + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1.0)

            print(loss_adv.item())

        import torchvision
        torch.save(x_adv, "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1.pt")
        torchvision.utils.save_image(x_adv[0], "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1.png")
        torchvision.utils.save_image(style[0], "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1_org.png")
        
        # return  style_mean, style_std
        input("attack done")
        return x_adv

    def adv_target_forward(self, content, style, adv):
        x_adv = style.detach() + 0.001 * torch.randn(style.shape).cuda().detach()
        # import ipdb
        # ipdb.set_trace()
        epsilon = 8 / 255.0
        alpha = 0.8 / 255.0
        # non-target
        target_feats = self.encode_with_intermediate(adv)

        for _step in range(50):
            print(_step)
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_feats = self.encode_with_intermediate(x_adv)
                # import ipdb
                # ipdb.set_trace()
                
                # adv_mean, adv_std = calc_mean_std(adv_feats[1])
                # target_mean, target_std = calc_mean_std(style_feats[1])
                loss_adv = 0

                for i in range(0, 4):
                    adv_mean, adv_std = calc_mean_std(adv_feats[i])
                    target_mean, target_std = calc_mean_std(target_feats[i])

                    loss_adv += self.calc_adv_loss(target_mean.detach(), target_std.detach(), adv_mean, adv_std)
                
            grad = torch.autograd.grad(loss_adv, [x_adv])[0]
            x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())

            x_adv = torch.min(torch.max(x_adv, style - epsilon), style + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1.0)

            print(loss_adv.item())

        import torchvision
        torch.save(x_adv, "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1_tgt.pt")
        torchvision.utils.save_image(x_adv[0], "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1_tgt.png")
        torchvision.utils.save_image(style[0], "/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/results/demo/attack1_org.png")
        
        # return  style_mean, style_std
        input("attack done")
        return x_adv
