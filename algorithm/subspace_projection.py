import torch


class Projection():
    def __init__(self, shot=5):
        self.num_dim = shot

    def hyperplanes(self, supportset_features, class_size, sample_size):
        all_hyper_planes = []
        means = []

        for ii in range(class_size):
            all_support_within_class_t = supportset_features[ii]
            meann = torch.mean(supportset_features[ii], dim=0)
            means.append(meann)
            all_support_within_class_t = all_support_within_class_t -  meann.unsqueeze(0).repeat(sample_size, 1)
            all_support_within_class = torch.transpose(all_support_within_class_t, 0, 1)
            uu, s, v = torch.svd(all_support_within_class, some=False)
            all_hyper_planes.append(uu[:, :self.num_dim])

        all_hyper_planes = torch.stack(all_hyper_planes, dim=0)
        means = torch.stack(means)

        return all_hyper_planes, means


    def projection_metric(self, target_features, hyperplanes, mu):
        batch_size = target_features.shape[0]
        class_size = hyperplanes.shape[0]
        similarities = []

        for j in range(class_size):
            h_plane_j = torch.squeeze(hyperplanes[j]).unsqueeze(0).repeat(batch_size, 1, 1)
            target_features_expanded = (target_features - mu[j].expand_as(target_features)).unsqueeze(-1)
            projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), target_features_expanded))
            projected_query_j = torch.squeeze(projected_query_j) + mu[j].unsqueeze(0).repeat(batch_size, 1)
            projected_query_dist_inter = target_features - projected_query_j
            query_loss = -torch.sum(projected_query_dist_inter*projected_query_dist_inter, dim=-1)
            similarities.append(query_loss)

        similarities = torch.stack(similarities, dim=1)


        return similarities