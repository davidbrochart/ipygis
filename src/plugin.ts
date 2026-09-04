import { Application, IPlugin } from '@lumino/application';

import { Widget } from '@lumino/widgets';
import type { ServiceManager } from '@jupyterlab/services';

import { IJupyterWidgetRegistry } from '@jupyter-widgets/base';

import * as widgetExports from './widget';

import { MODULE_NAME, MODULE_VERSION } from './version';

type GISApplication = Application<Widget> & {
  serviceManager: ServiceManager.IManager;
};

const EXTENSION_ID = 'ipygis:plugin';

/**
 * The gis plugin.
 */
const gisPlugin: IPlugin<GISApplication, void> = {
  id: EXTENSION_ID,
  requires: [IJupyterWidgetRegistry],
  activate: activateWidgetExtension,
  autoStart: true,
} as unknown as IPlugin<GISApplication, void>;
// the "as unknown as ..." typecast above is solely to support JupyterLab 1
// and 2 in the same codebase and should be removed when we migrate to Lumino.

export default gisPlugin;

/**
 * Activate the widget extension.
 */
function activateWidgetExtension(
  app: GISApplication,
  registry: IJupyterWidgetRegistry,
): void {
  widgetExports.GISModel.contentsManager = app.serviceManager.contents;
  registry.registerWidget({
    name: MODULE_NAME,
    version: MODULE_VERSION,
    exports: widgetExports,
  });
}
