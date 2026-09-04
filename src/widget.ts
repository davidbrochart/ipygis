import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';

import type { Contents } from '@jupyterlab/services';

import { IcechunkBridge } from './icechunk';
import { MODULE_NAME, MODULE_VERSION } from './version';

export class GISModel extends DOMWidgetModel {
  static contentsManager: Contents.IManager | undefined;

  private bridge!: IcechunkBridge;

  initialize(...args: Parameters<DOMWidgetModel['initialize']>) {
    super.initialize(...args);
    this.bridge = new IcechunkBridge(this, GISModel.contentsManager);
    this.send({ type: 'icechunk_ready' }, this.callbacks());
  }

  async close(commClosed = false) {
    this.bridge.dispose();
    await super.close(commClosed);
  }

  defaults() {
    return {
      ...super.defaults(),
      _model_name: GISModel.model_name,
      _model_module: GISModel.model_module,
      _model_module_version: GISModel.model_module_version,
      _view_name: GISModel.view_name,
      _view_module: GISModel.view_module,
      _view_module_version: GISModel.view_module_version,
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
  };

  static model_name = 'GISModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'GISView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class GISView extends DOMWidgetView {}
