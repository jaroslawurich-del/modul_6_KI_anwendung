#!/usr/bin/env python
import sys

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

# Windows consoles default to cp1252, which can't print some characters crewAI's
# verbose logging and LLM output use (emoji, narrow no-break spaces). Force UTF-8
# so those don't crash the process.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tech_support_crew.crews.tech_support_crew.tech_support_crew import TechSupportCrew

# Swap this ticket to see the manager route to a different specialist:
#
# Billing:   "I was charged twice for my subscription this month and I want a
#             refund for the duplicate charge."
# Technical: "The app crashes with a 500 error every time I try to export a
#             report as PDF."
# Account:   "I'm locked out of my account after changing my email address
#             and the reset link never arrives."
SAMPLE_TICKET = (
    "I was charged twice for my subscription this month and I want a refund "
    "for the duplicate charge."
)


class TicketState(BaseModel):
    ticket: str = SAMPLE_TICKET
    resolution: str = ""


class TechSupportFlow(Flow[TicketState]):

    @start()
    def receive_ticket(self):
        print(f"New ticket received:\n{self.state.ticket}\n")

    @listen(receive_ticket)
    def route_and_resolve(self):
        result = (
            TechSupportCrew()
            .crew()
            .kickoff(inputs={"ticket": self.state.ticket})
        )
        self.state.resolution = result.raw

    @listen(route_and_resolve)
    def deliver_resolution(self):
        with open("resolution.txt", "w", encoding="utf-8") as f:
            f.write(self.state.resolution)
        print("\nFinal resolution:\n")
        print(self.state.resolution)


def kickoff():
    flow = TechSupportFlow()
    flow.kickoff()


def plot():
    flow = TechSupportFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
