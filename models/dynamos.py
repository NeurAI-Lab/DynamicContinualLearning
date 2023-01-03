from utils.buffer import Buffer
from torch.nn import functional as F
from models.dynamos_sgd import DynamosSgd
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Filtered Convolutional Policy.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    add_policy_args(parser)
    return parser


class Dynamos(DynamosSgd):
    NAME = 'dynamos'
    COMPATIBILITY = ['class-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Dynamos, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, policy=True,
                             add_if_correct=True)

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

        if not self.net.get_warmup() and not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, buf_policy_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
            buf_outputs, buf_actions, buf_policy_outputs, buf_log_probs = self.net(buf_inputs)

            mse_main_loss = F.mse_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * mse_main_loss
            if self.args.policy_alpha:
                mse_policy_loss = F.mse_loss(buf_policy_outputs, buf_policy_logits)
                loss += self.args.policy_alpha * mse_policy_loss
                losses.update({
                    "Logits_Policy_buff": mse_policy_loss.item(),
                })

            buf_rewards = self.get_reward(buf_outputs, buf_actions, buf_labels)
            main_buf_loss, policy_buf_loss = self.loss(buf_outputs, buf_labels, buf_rewards, buf_log_probs)

            loss += self.args.beta * (main_buf_loss + self.args.reward_weight * policy_buf_loss)

            if self.args.prototype_loss:
                correct_locs = self.get_correct_locs(buf_outputs, buf_labels)
                if correct_locs.sum() > 2:
                    buf_prototype_loss = self.get_prototype_loss(buf_policy_outputs[correct_locs],
                                                                 buf_labels[correct_locs])
                    loss += self.args.prototype_loss * buf_prototype_loss
                    losses.update({
                        "Buf_Prototype_loss": buf_prototype_loss.item()
                    })

            losses.update(
                {
                    "Total_loss": loss.item(),
                    "Logits_buff": mse_main_loss.item(),
                    "CE_buff": main_buf_loss.item(),
                    "Rwd_buff": policy_buf_loss.item(),
                }
            )

        loss.backward()
        self.opt.step()

        if not self.net.get_warmup():
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data,
                                 policy_logits=policy_logits.data)

        return losses