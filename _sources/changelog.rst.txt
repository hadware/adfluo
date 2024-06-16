=========
Changelog
=========

0.4.0 - 16/06/2024
==================

- Dropped support for python < 3.10
- Added support for 3.12
- Completely migrated to "3.10-compatible" type-hinting

0.3.7 - 15/06/2024
==================

- Added option to flatten features composed of dicts (or nested dicts) when storing
- Added experimental "default" option for features,
  used to return a default value when something goes wrong during the extraction of that feature for a sample.