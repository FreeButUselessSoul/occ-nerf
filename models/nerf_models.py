import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class OfficialNerf(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Sequential(nn.Linear(D, 1),nn.Softplus())
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Sequential(nn.Linear(D//2, 3))

        # self.fc_density[0].bias.data = torch.tensor([0.1]).float()
        # self.fc_rgb[0].bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=3)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den


class fullNeRF(nn.Module):
    def __init__(self, in_channels_xyz, in_channels_dir, W, D=8, skips=[4]):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir, W), nn.ReLU(True),
                        nn.Linear(W, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, W//2),nn.ReLU(inplace=True),
                                        nn.Linear(W//2, 3))

    def forward(self, input_xyz, input_dir_a):
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                # import ipdb;ipdb.set_trace()
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)

        static_rgb = self.static_rgb(dir_encoding)
        static = torch.cat([static_rgb, static_sigma], -1) # (B, 4)

        return static