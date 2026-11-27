# external-change-notification Specification

## Purpose
Keeps several app processes that share one database aware of each other's edits to family data, so the user sees and can refresh stale views instead of working on outdated information.

## Requirements

### Requirement: Record family-data writes per process
Every insert, update and delete that an app process makes to persons, relationships, custom field definitions, custom field values or saved tree views SHALL be recorded with the table, the row identifier and an identifier of the writing process.

#### Scenario: Person renamed
- **WHEN** process A renames person 12
- **THEN** a change entry exists for the persons table, row 12, written by process A

#### Scenario: Unwatched table
- **WHEN** process A assigns a face to a person without changing the person row
- **THEN** no change entry is recorded for that write

### Requirement: Detect other processes' changes on activation
When an app window becomes active, the app SHALL look for change entries written by other processes since its previous check. Entries written by the same process SHALL be ignored.

#### Scenario: Change in the other app
- **WHEN** Tree Studio edits person 12 while Face-Local is in the background, and Face-Local's window is then activated
- **THEN** Face-Local detects one changed person

#### Scenario: Own change
- **WHEN** Face-Local edits person 12 and its window is later re-activated with no other writes
- **THEN** no external change is detected

### Requirement: Notify and refresh on request
When external changes are detected and automatic refresh is off, the app SHALL show a non-blocking notice that summarises the changed data by kind and count, with a Refresh action. Refresh SHALL reload the affected views and hide the notice. Dismissing the notice SHALL leave the views as they are.

#### Scenario: Notice summary
- **WHEN** 3 persons and 1 relationship were changed by another process
- **THEN** the notice mentions 3 persons and 1 relationship
- **AND** the user can keep working without closing it

#### Scenario: Refresh
- **WHEN** the user clicks Refresh
- **THEN** the persons list and the family tree show the changed data
- **AND** the notice disappears

### Requirement: Automatic refresh setting
The app SHALL offer a setting to refresh automatically when the window is activated. It SHALL be off by default and SHALL be remembered between runs. When it is on, detected external changes SHALL refresh the affected views without showing the notice.

#### Scenario: Auto refresh on
- **WHEN** the setting is on and external changes are detected on activation
- **THEN** the affected views refresh with no notice shown

#### Scenario: Default
- **WHEN** the app runs for the first time after this change
- **THEN** the setting is off

### Requirement: Bounded change log
The change log SHALL be pruned automatically so that entries older than 30 days are removed, without affecting detection of changes that happened since any running process's last check.

#### Scenario: Prune old entries
- **WHEN** the app starts and the log has entries older than 30 days
- **THEN** those entries are removed

### Requirement: Single-process use is unaffected
With only one app process running, recording changes SHALL NOT show any notice and SHALL NOT noticeably slow down person or relationship edits.

#### Scenario: Only Face-Local running
- **WHEN** only Face-Local runs and the user edits, merges and deletes persons
- **THEN** no external-change notice ever appears
