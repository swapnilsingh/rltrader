# trainer/agent/__init__.py

from trainer.agent.dqn_trainer import DQNTrainer

if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.bootstrap()
    trainer.run_training_loop()

