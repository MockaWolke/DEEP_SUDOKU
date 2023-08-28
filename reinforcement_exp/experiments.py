from multi_action_architectures import *
import deepsudoku.reinforcement_learning
from single_action_architectures import Single_Action_MLP_Onehot, SingleConvActorOnehot, SingleOnlyConvActorOnehot, OnlyConvSeperateValue, OnlyConvSeperateValueBigger

Multi_Action_AGENTS = { 
'SplitMLP' : SplitMLP, 
'SplitMLPOnehot' : SplitMLPOnehot, 
'ConvActor' : ConvActor, 
'ConvActorOnehot' : ConvActorOnehot, }

Single_Action_Agents = {
    "Single_Action_MLP_Onehot" :Single_Action_MLP_Onehot,
    "SingleConvActorOnehot" :SingleConvActorOnehot,
    "SingleOnlyConvActorOnehot": SingleOnlyConvActorOnehot,
    "OnlyConvSeperateValue" :OnlyConvSeperateValue,
    "OnlyConvSeperateValueBigger" :OnlyConvSeperateValueBigger,
}
    
