"""Candidate command dispatcher; no paid generation is started here."""
import runpy
import sys


def main():
    actions={
        'containers':'containers', 'extend-containers':'container_extension',
        'compose':'composition', 'graspgenx':'graspgenx', 'convert':'conversion',
        'export':'condition_export_v2',
    }
    if len(sys.argv)<2 or sys.argv[1] in {'-h','--help'}:
        print('General pick/place candidate (API budget 0).\n'
              'Commands: init, run, audit, containers, extend-containers, compose, graspgenx, convert, export.\n'
              'Use COMMAND --help for arguments. Physics uses an idle locked GPU and a shared finite budget.')
        return
    command=sys.argv[1]
    if command in actions:
        name=actions[command];sys.argv=[sys.argv[0],*sys.argv[2:]]
    elif command in {'init','run','audit','worker'}:
        name='runner'
    else:raise SystemExit('Unknown pickplace command: '+command)
    runpy.run_module('resimpair_pickplace.'+name,run_name='__main__')


if __name__=='__main__':main()
