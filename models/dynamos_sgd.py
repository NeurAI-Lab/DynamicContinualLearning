from utils.args import *
from models.utils.continual_model import ContinualModel
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_policy_args(parser)
    return parser


class DynamosSgd(ContinualModel):
    NAME = 'dynamos_sgd'
    COMPATIBILITY = ['class-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DynamosSgd, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        if not self.net.get_warmup():
            outputs, actions, policy_logits, log_probs = self.net(inputs)
            rewards = self.get_reward(outputs, actions, labels)
            main_loss, policy_loss = self.loss(outputs, labels, rewards, log_probs)
            loss = main_loss + self.args.reward_weight * policy_loss
            losses = {
                "CE_curr": main_loss.item(),
                "Rwd_curr": policy_loss.item(),
            }
            if self.args.prototype_loss:
                correct_locs = self.get_correct_locs(outputs, labels)
                if correct_locs.sum() > 2:
                    prototype_loss = self.get_prototype_loss(policy_logits[correct_locs], labels[correct_locs])
                    loss += self.args.prototype_loss * prototype_loss
                    losses.update({
                        "Total_loss": loss.item(),
                        "Prototype_loss": prototype_loss.item()
                    })
            losses.update({
                "Total_loss": loss.item()
            })
        else:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            losses = {
                "Total_loss": loss.item(),
            }
        loss.backward()
        self.opt.step()

        return losses

    def get_reward(self, outputs, actions, labels):
        _, preds_idx = outputs.max(1)
        rewards_all = actions
        rewards = torch.ones((outputs.shape[0], len(self.net.agent_boundaries) - 1), device=self.device)
        for j, agent_boundary in enumerate(self.net.agent_boundaries[:-1]):
            diff = self.args.keep_ratio - \
                             rewards_all[:, agent_boundary: self.net.agent_boundaries[j + 1]].mean(dim=1)
            rewards[:, j] = -diff ** 2

        rewards[preds_idx != labels] *= self.args.policy_penalty * -1.0
        return rewards

    def get_prototype_loss(self, logits, labels):
        x_square = logits.pow(2).sum(dim=-1)
        prod = torch.matmul(logits, logits.t())
        distance = torch.unsqueeze(x_square, 1) + torch.unsqueeze(x_square, 0) - 2 * prod
        distance = torch.triu(distance, diagonal=1)
        mask_n = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float()
        num = (distance * mask_n).sum() / logits.shape[1]
        mask_d = torch.ne(labels.unsqueeze(1), labels.unsqueeze(1).T).float()
        den = (distance * mask_d).sum() / logits.shape[1]
        return (1 + num) / (1 + den)

    def get_correct_locs(self, outputs, labels):
        _, preds_idx = outputs.max(1)
        correct_locs = preds_idx == labels
        return correct_locs

    def forward(self, x, threshold=0.5):
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        if self.net.get_warmup():
            results = self.net(x, threshold=threshold)
            return results
        else:
            results, actions = self.net(x, threshold=threshold)
            return results
