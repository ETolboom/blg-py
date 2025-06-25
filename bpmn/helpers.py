from typing import List

def get_missing_tasks(source_tasks: List[str], target_tasks: List[str]) -> list[str]:
    """Helper function that returns a list of missing tasks in the source model when compared to a target model"""
    missing_tasks: list[str] = []

    for target_task in target_tasks:
        missing_task = True
        for source_task in source_tasks:
            if target_task.lower() == source_task.lower():
                missing_task = False
                break
        if missing_task:
            missing_tasks.append(target_task)

    return missing_tasks
