from aiogram.filters import Command

from tgbot.handlers import start


def test_cmd_help_uses_command_filter() -> None:
    help_handler = next(
        handler
        for handler in start.router.message.handlers
        if handler.callback.__name__ == "cmd_help"
    )

    command_filters = [
        filter_obj.callback
        for filter_obj in help_handler.filters
        if isinstance(filter_obj.callback, Command)
    ]

    assert command_filters
    assert command_filters[0].commands == ("help",)
