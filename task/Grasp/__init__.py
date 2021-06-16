from gym.envs.registration import register
from task.Grasp.PandaGraspEnv import PandaGraspEnv
from task.Reach.PandaReachEnv import PandaReachEnv


register(
	id='PandaGraspEnv-v0',
	entry_point='task.Grasp.PandaGraspEnv:PandaGraspEnv',
	kwargs={
		
	}
)
