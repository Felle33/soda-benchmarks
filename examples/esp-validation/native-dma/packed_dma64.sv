// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Transport ABI 2: whole 64-bit beats, beat-index/count requests.
// No payload buffers, lane conversion, speculative DMA or admission policy.
`timescale 1ns/1ps
module sqr_bambu_basic_dma64 #(
  parameter CHECK_ARCHIVE_GEMM=0, parameter TIMEOUT_CYCLES=1000000,
  parameter DMA_WIDTH=64
)(
  input clk,rst,conf_done, output reg acc_done, output [31:0] debug,
  output dma_read_ctrl_valid,
  output [31:0] dma_read_ctrl_data_index,dma_read_ctrl_data_length,
  output [2:0] dma_read_ctrl_data_size, output [5:0] dma_read_ctrl_data_user,
  input dma_read_ctrl_ready, input dma_read_chnl_valid,
  input [63:0] dma_read_chnl_data, output dma_read_chnl_ready,
  output dma_write_ctrl_valid,
  output [31:0] dma_write_ctrl_data_index,dma_write_ctrl_data_length,
  output [2:0] dma_write_ctrl_data_size, output [5:0] dma_write_ctrl_data_user,
  input dma_write_ctrl_ready, output dma_write_chnl_valid,
  output [63:0] dma_write_chnl_data, input dma_write_chnl_ready
);
  // Deliberate elaboration error for an unsupported physical configuration,
  // including when simulation assertions are compiled out for synthesis.
  // Older Verilator checks the placeholder module before pruning this branch.
  /* verilator lint_off TIMESCALEMOD */
  generate if (DMA_WIDTH != 64) begin : unsupported_dma_width
    SODAP_PACKED_DMA_REQUIRES_WIDTH_64 invalid_configuration();
  end endgenerate
  /* verilator lint_on TIMESCALEMOD */
  wire [63:0] rr,wr,wd;
  wire rr_valid,rr_ready,wr_valid,wr_ready,rd_ready,wd_valid,wd_ready,core_done;
  reg core_start,done_pending,read_active,write_active;
  reg [31:0] read_left,write_left;
  always @(posedge clk) if(!rst) core_start<=0; else core_start<=conf_done;
  forward_kernel u_core(
    .clock(clk),.reset(rst),.start_port(core_start),.done_port(core_done),.idle_port(),
    .P0_TDATA(rr),.P0_TVALID(rr_valid),.P0_TREADY(rr_ready),
    .P1_TDATA(dma_read_chnl_data),.P1_TVALID(dma_read_chnl_valid&&read_active),.P1_TREADY(rd_ready),
    .P2_TDATA(wr),.P2_TVALID(wr_valid),.P2_TREADY(wr_ready),
    .P3_TDATA(wd),.P3_TVALID(wd_valid),.P3_TREADY(wd_ready));
  wire allow_write;
`ifdef SYNTHESIS
  assign allow_write=1'b1;
`else
  integer stall_until=0,cycles=0;
  initial begin
    if(DMA_WIDTH!=64) $fatal(1,"packed Bambu backend requires CONFIG_DMA_NOC_WIDTH=64");
    if(!$value$plusargs("gemm_stall=%d",stall_until)) stall_until=0;
  end
  always @(posedge clk) if(!rst||conf_done) cycles<=0; else cycles<=cycles+1;
  assign allow_write=cycles>=stall_until;
`endif
  assign dma_read_ctrl_valid=rr_valid&&!read_active;
  assign rr_ready=dma_read_ctrl_ready&&!read_active;
  assign dma_read_ctrl_data_index=rr[31:0];
  assign dma_read_ctrl_data_length=rr[63:32];
  assign dma_read_ctrl_data_size=3'd3;
  assign dma_read_ctrl_data_user=0;
  assign dma_read_chnl_ready=rd_ready&&read_active;
  assign dma_write_ctrl_valid=wr_valid&&!write_active&&allow_write;
  assign wr_ready=dma_write_ctrl_ready&&!write_active&&allow_write;
  assign dma_write_ctrl_data_index=wr[31:0];
  assign dma_write_ctrl_data_length=wr[63:32];
  assign dma_write_ctrl_data_size=3'd3;
  assign dma_write_ctrl_data_user=0;
  assign dma_write_chnl_data=wd;
  assign dma_write_chnl_valid=wd_valid&&write_active;
  assign wd_ready=dma_write_chnl_ready&&write_active;
  assign debug={29'b0,done_pending,write_active,read_active};
  always @(posedge clk) begin
    if(!rst) begin
      acc_done<=0;done_pending<=0;read_active<=0;write_active<=0;read_left<=0;write_left<=0;
    end else begin
      acc_done<=0;
      if(dma_read_ctrl_valid&&dma_read_ctrl_ready) begin read_active<=1;read_left<=rr[63:32];end
      if(dma_read_chnl_valid&&dma_read_chnl_ready) begin
        read_left<=read_left-1;if(read_left==1)read_active<=0;
      end
      if(dma_write_ctrl_valid&&dma_write_ctrl_ready) begin write_active<=1;write_left<=wr[63:32];end
      if(dma_write_chnl_valid&&dma_write_chnl_ready) begin
        write_left<=write_left-1;if(write_left==1)write_active<=0;
      end
      if(core_done)done_pending<=1;
      if(done_pending&&!read_active&&!write_active&&!rr_valid&&!wr_valid&&!wd_valid) begin
        acc_done<=1;done_pending<=0;
      end
    end
  end
  // synthesis translate_off
`ifndef SYNTHESIS
  reg rr_held=0,wr_held=0,wd_held=0;
  reg [63:0] prev_rr,prev_wr,prev_wd;
  integer invocation=0,nr=0,nw=0,read_beats=0,write_beats=0;
  reg running=0;
  always @(posedge clk) begin
    if(!rst) begin rr_held=0;wr_held=0;wd_held=0;running=0;end
    else begin
      if(conf_done)begin invocation++;running=1;nr=0;nw=0;read_beats=0;write_beats=0;end
      if(running&&cycles==TIMEOUT_CYCLES)$fatal(1,"packed DMA timeout");
      if(rr_held&&(!dma_read_ctrl_valid||rr!==prev_rr))$fatal(1,"read request changed under stall");
      if(wr_held&&(!dma_write_ctrl_valid||wr!==prev_wr))$fatal(1,"write request changed under stall");
      if(wd_held&&(!dma_write_chnl_valid||wd!==prev_wd))$fatal(1,"write data changed under stall");
      rr_held=dma_read_ctrl_valid&&!dma_read_ctrl_ready;prev_rr=rr;
      wr_held=dma_write_ctrl_valid&&!dma_write_ctrl_ready;prev_wr=wr;
      wd_held=dma_write_chnl_valid&&!dma_write_chnl_ready;prev_wd=wd;
      if(dma_read_ctrl_valid&&dma_read_ctrl_ready)begin nr++;if(!rr[63:32])$fatal(1,"empty read");end
      if(dma_write_ctrl_valid&&dma_write_ctrl_ready)begin nw++;if(!wr[63:32])$fatal(1,"empty write");end
      if(dma_read_chnl_valid&&dma_read_chnl_ready)read_beats++;
      if(dma_write_chnl_valid&&dma_write_chnl_ready)write_beats++;
      if(acc_done)begin
        running=0;
        $display("THIN_BOUNDARY RESULT invocation=%0d requests=%0d/%0d beats=%0d/%0d cycles=%0d",invocation,nr,nw,read_beats,write_beats,cycles);
      end
    end
  end
`endif
  // synthesis translate_on
endmodule
