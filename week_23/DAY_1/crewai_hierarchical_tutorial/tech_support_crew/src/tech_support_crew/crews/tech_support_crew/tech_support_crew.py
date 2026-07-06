from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class TechSupportCrew:
    """Hierarchical tech-support escalation crew.

    The manager (support_lead) is defined but deliberately NOT decorated with
    @agent, so it stays out of the crew's worker roster and is only wired in
    as `manager_agent`. The single task below has no `agent:` assigned in
    tasks.yaml, so the manager decides which specialist handles it based on
    the ticket content, then reviews the draft before accepting it.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def support_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["support_lead"],  # type: ignore[index]
        )

    @agent
    def billing_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["billing_specialist"],  # type: ignore[index]
        )

    @agent
    def technical_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_specialist"],  # type: ignore[index]
        )

    @agent
    def account_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["account_specialist"],  # type: ignore[index]
        )

    @task
    def resolve_ticket_task(self) -> Task:
        return Task(
            config=self.tasks_config["resolve_ticket_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Tech Support Escalation Crew"""
        return Crew(
            agents=self.agents,  # billing, technical, account specialists only
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.support_lead(),
            verbose=True,
        )
