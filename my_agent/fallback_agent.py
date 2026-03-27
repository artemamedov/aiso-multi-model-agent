from contextlib import aclosing
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event

from .modeling import should_fallback_on_error


class ProviderFallbackAgent(BaseAgent):
    """Run a primary agent and retry with a fallback agent on provider failures."""

    primary_agent: BaseAgent
    fallback_agent: BaseAgent

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        async for event in self._run_with_fallback(ctx, live=False):
            yield event

    async def _run_live_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        async for event in self._run_with_fallback(ctx, live=True):
            yield event

    async def _run_with_fallback(
        self,
        ctx: InvocationContext,
        *,
        live: bool,
    ) -> AsyncGenerator[Event, None]:
        runner = self.primary_agent.run_live if live else self.primary_agent.run_async
        try:
            async with aclosing(runner(ctx)) as agen:
                async for event in agen:
                    yield event
            return
        except Exception as error:
            if not should_fallback_on_error(error):
                raise

        runner = self.fallback_agent.run_live if live else self.fallback_agent.run_async
        async with aclosing(runner(ctx)) as agen:
            async for event in agen:
                yield event



class DeterministicRouter(BaseAgent):
    """Route requests to specialist agents using generic modality classification."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        target = self._resolve_target_agent(ctx)
        async with aclosing(target.run_async(ctx)) as agen:
            async for event in agen:
                yield event

    async def _run_live_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        target = self._resolve_target_agent(ctx)
        async with aclosing(target.run_live(ctx)) as agen:
            async for event in agen:
                yield event

    def _resolve_target_agent(self, ctx: InvocationContext) -> BaseAgent:
        from .routing import classify_request, extract_user_text

        user_text = extract_user_text(ctx)
        target_name = classify_request(user_text)
        target = next((agent for agent in self.sub_agents if agent.name == target_name), None)
        if target is None:
            raise ValueError(f"Could not find specialist agent: {target_name}")
        return target
