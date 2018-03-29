# Contributing Guidelines

Our main development happens on the `master` branch and is continuously
verified for correctness. If you would like to contribute your work to the bml
project, please follow the instructions at the GitHub help page ["About pull
requests"](https://help.github.com/articles/about-pull-requests/). To
summarize:

- Fork the project on github
- Clone that forked repository
- Create a branch in it
- Commit any changes to the branch
- Push the branch to your forked repository
- Go to https://github.com/lanl/bml and click on 'Create Pull Request'

During the review process you might want to update your pull
request. Please add commits or `amend` your existing commits as
necessary. If you amend any commits you need to add the
`--force-with-lease` option to the `git push` command. Please make
sure that your pull request contains only one logical change (see
["Structural split of
change"](https://wiki.openstack.org/wiki/GitCommitMessages#Structural_split_of_changes)
for further details.

## Coding Style

Please indent your C code using

    $ indent -gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda

You can use the script `indent.sh` to indent all C code.
