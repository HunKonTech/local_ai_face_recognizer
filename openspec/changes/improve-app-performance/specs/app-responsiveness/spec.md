# Spec Delta

## Purpose

Keeps Face-Local responsive on large family archives by setting measurable response-time budgets for everyday operations and forbidding UI-thread stalls caused by bulk work.

## ADDED Requirements

### Requirement: Reference benchmark
The project SHALL provide a benchmark that measures start-up, the first display of every main tab, the persons refresh and deleting 400 persons on a copy of a given database, without modifying that database, and reports each value next to its budget.

#### Scenario: Run the benchmark
- **WHEN** the benchmark is run with a path to a database
- **THEN** it works on a copy, prints each measured time and its budget, and the original database file is unchanged

### Requirement: Start-up budget
On the reference database (≈1,600 persons, ≈23,000 faces, ≈13,000 images) the main window SHALL become interactive within 6 seconds of process start on the reference Windows machine, and no panel SHALL decode original photos during start-up.

#### Scenario: Start-up with many objects
- **WHEN** the app starts on a database with tagged objects whose photos are large
- **THEN** the window is interactive within the budget
- **AND** object thumbnails are produced after start-up, off the UI thread

### Requirement: Tab switching budget
Showing any main tab for the first time SHALL take at most 0.5 seconds of UI-thread time on the reference database; images and thumbnails MAY continue to appear afterwards.

#### Scenario: Persons tab
- **WHEN** the user opens the Persons tab with 1,556 persons
- **THEN** the table is shown within 0.5 seconds and thumbnails fill in progressively

### Requirement: Person deletion and merge budget
Deleting or merging persons SHALL NOT scan whole tables per deleted person. Deleting 400 persons in one operation SHALL complete within 0.5 seconds on the reference database. Every foreign-key column SHALL have an index whose first column is that column.

#### Scenario: Cleanup of empty Unknown persons
- **WHEN** 400 empty Unknown persons are cleaned up
- **THEN** the operation completes within 0.5 seconds

#### Scenario: New foreign key without index
- **WHEN** a developer adds a foreign-key column without an index
- **THEN** the automated test suite fails and names the column

### Requirement: No UI-thread stalls from bulk work
Thumbnail decoding, file-existence checks over many files and full persons reloads SHALL NOT run on the UI thread. A single person edit SHALL update the visible person lists within 0.2 seconds of UI-thread time.

#### Scenario: Rename one person
- **WHEN** the user renames a person on a database with 1,556 persons
- **THEN** the sidebar and the Persons tab show the new name without a full rebuild blocking the UI for more than 0.2 seconds

### Requirement: Reused thumbnails
A thumbnail that was produced once for a given file, size and crop SHALL be reused across panels and across app restarts until the file changes.

#### Scenario: Second start
- **WHEN** the app is started a second time without changes to the photos
- **THEN** person and object thumbnails are read from the cache instead of being decoded from the originals
