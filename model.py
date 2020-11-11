import torch

import utils


class BuddingTree(torch.nn.Module):

    def __init__(self, in_features, out_features, max_depth, tau):
        super(BuddingTree, self).__init__()
        assert max_depth > 1, "max_depth should be larger than 1."
        self.in_features = in_features
        self.out_features = out_features
        self.max_depth = max_depth
        self.tau = tau
        self.prior = torch.tensor(0.5)

        self.gamma = torch.nn.ParameterDict({
            "0": torch.nn.Parameter(torch.tensor(1.)),
        })
        self.node_max = 0
        self.gate_weight = {}
        self.gate_bias = {}
        self.leaf_weight = {}
        self.leaf_bias = {}
        gate_weight = torch.nn.init.xavier_normal_(torch.empty(1, in_features))
        self.gate_weight["0"] = torch.nn.Parameter(gate_weight.reshape(-1))
        self.gate_bias["0"] = torch.nn.Parameter(torch.tensor([0.]))

        leaf_weight = torch.nn.init.xavier_normal_(torch.empty(out_features, in_features))
        self.leaf_weight["0"] = torch.nn.Parameter(leaf_weight.T)
        self.leaf_bias["0"] = torch.nn.Parameter(torch.zeros(out_features))
        self._create_node("1")
        self._create_node("2")

        self.gate_weight = torch.nn.ParameterDict(self.gate_weight)
        self.gate_bias = torch.nn.ParameterDict(self.gate_bias)
        self.leaf_weight = torch.nn.ParameterDict(self.leaf_weight)
        self.leaf_bias = torch.nn.ParameterDict(self.leaf_bias)

    def forward(self, x):
        return self.node_forward(x, "0")

    def gate_forward(self, x, select=None):
        if select is None:
            select = self.gamma.keys()

        w = torch.stack([self.gate_weight[i] for i in select], dim=-1)
        b = torch.stack([self.gate_bias[i] for i in select], dim=-1)
        gatings = x @ w + b
        named_gatings = {}
        for i, idx in enumerate(select):
            named_gatings[idx] = gatings[:, i].reshape(-1, 1)
        return named_gatings

    def leaf_forward(self, x, select=None):
        if select is None:
            select = self.gamma.keys()

        w = torch.stack([self.leaf_weight[i] for i in select])
        b = torch.stack([self.leaf_bias[i] for i in select]).unsqueeze(1)
        responses = (x @ w) + b
        named_responses = {}
        for i, idx in enumerate(select):
            named_responses[idx] = responses[i]
        return named_responses

    def node_forward(self, x, idx, gatings=None, responses=None):
        bud_list = self.get_buds(idx)

        if responses is None:
            filtered = [i for (i, _) in bud_list]
            responses = self.leaf_forward(x, filtered)

        if self.is_bud(idx):
            # print("idx: %d -> bud computation" % idx)
            if gatings is None:
                filtered = [i for (i, is_bud) in bud_list if is_bud]
                gatings = self.gate_forward(x, filtered)
            gamma = self.gamma[idx].repeat(x.shape[0], 1)
            gamma_z = utils.gumbel_sigmoid(gamma, hard=True)
            gate = gatings[idx]
            gate_z = utils.gumbel_sigmoid(gate, hard=True)
            left_response = self.node_forward(x, utils.left_child_idx(idx), gatings, responses)
            right_response = self.node_forward(x, utils.right_child_idx(idx), gatings, responses)
            m = (1-gamma_z) * responses[idx] + gamma_z * (gate_z * left_response + (1-gate_z) * right_response)
            return m
        else:
            # print("idx: %d -> leaf computation" % idx)
            return responses[idx]

    def get_buds(self, idx):
        if idx not in self.gamma:
            print("Node %s is not in the tree." % idx)
            return []

        bud_list = []
        left_bud_list = []
        right_bud_list = []

        bud_list.append((idx, self.is_bud(idx)))
        if utils.left_child_idx(idx) in self.gamma:
            left_bud_list = self.get_buds(utils.left_child_idx(idx))

        if utils.right_child_idx(idx) in self.gamma:
            right_bud_list = self.get_buds(utils.right_child_idx(idx))

        return bud_list + left_bud_list + right_bud_list

    def is_bud(self, idx):
        with torch.no_grad():
            if idx not in self.gamma:
                print("Node %s is not in the tree." % idx)
                return

            if idx == "0":
                return True
            else:
                parent_idx = utils.parent_idx(idx)
                if torch.sigmoid(self.gamma[parent_idx]) > self.tau:
                    return True
                else:
                    return False

    def update_nodes(self):
        with torch.no_grad():
            changed = False
            current_nodes = list(self.gamma.keys())
            for key in current_nodes:
                if torch.sigmoid(self.gamma[key]) > self.tau:
                    grandchildren = [
                        utils.left_child_idx(utils.left_child_idx(key)),
                        utils.right_child_idx(utils.left_child_idx(key)),
                        utils.left_child_idx(utils.right_child_idx(key)),
                        utils.right_child_idx(utils.right_child_idx(key))]
                    for gc in grandchildren:
                        if gc not in self.gamma:
                            changed = True
                            self._create_node(gc)

            return changed

    def kl_loss(self):
        loss = 0.0
        for key in self.gamma:
            if self.is_bud(key):
                q = torch.distributions.Bernoulli(torch.sigmoid(self.gamma[key]))
                p = torch.distributions.Bernoulli(self.prior)
                loss += torch.distributions.kl_divergence(q, p)
        return loss

    def clamp_gamma(self):
        with torch.no_grad():
            for key in self.gamma:
                self.gamma[key].clamp_(1e-8, 1.)

    def _create_node(self, idx):
        if idx in self.gamma:
            print("Node %s is already in the tree." % idx)
            return False

        parent = utils.parent_idx(idx)
        parent_gw = self.gate_weight[parent]
        parent_gb = self.gate_bias[parent]
        parent_lw = self.leaf_weight[parent]
        parent_lb = self.leaf_bias[parent]

        self.gamma[idx] = torch.nn.Parameter(torch.tensor(1.))
        self.gate_weight[idx] = torch.nn.Parameter(utils.return_perturbed(parent_gw, 0.95, 1.05))
        self.gate_bias[idx] = torch.nn.Parameter(utils.return_perturbed(parent_gb, 0.95, 1.05))
        self.leaf_weight[idx] = torch.nn.Parameter(utils.return_perturbed(parent_lw, 0.95, 1.05))
        self.leaf_bias[idx] = torch.nn.Parameter(utils.return_perturbed(parent_lb, 0.95, 1.05))

        if int(idx) > self.node_max:
            self.node_max = int(idx)
        print("Node %s is initialized." % idx)

    def print_tree(self, node="0", string=""):
        if node in self.gamma:
            print("|%s(%s): %.3f grad:" % (string, node, torch.sigmoid(self.gamma[node]).item()), self.gamma[node].grad)
            self.print_tree(utils.left_child_idx(node), string+"----")
            self.print_tree(utils.right_child_idx(node), string+"----")
